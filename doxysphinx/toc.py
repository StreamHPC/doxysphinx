# =====================================================================================
#  C O P Y R I G H T
# -------------------------------------------------------------------------------------
#  Copyright (c) 2023 by Robert Bosch GmbH. All rights reserved.
#
#  Author(s):
#  - Markus Braun, :em engineering methods AG (contracted by Robert Bosch GmbH)
# =====================================================================================
"""The toc module contains classes related to the toctree generation for doxygen htmls/rsts."""

import logging
import operator
import re
import unicodedata
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
)
from xml.etree.ElementTree import (
    Element as XMLElement,  # nosec: B405 #https://github.com/PyCQA/bandit/issues/709
)

import defusedxml.ElementTree
from packaging.version import Version

from doxysphinx.doxygen import read_js_data_file
from doxysphinx.utils.files import write_file
from doxysphinx.utils.iterators import apply

_logger = logging.getLogger()


class TocGenerator(Protocol):
    """
    TocGenerator protocol.

    Gets the source_dir (with the html sources) during init and
    each file to possibly generate a toctree directive for in the :meth:`generate_toc_for`
    method. The implementer has then to choose how to implement the toc generation.
    """

    def __init__(self, source_dir: Path, tagfile: Optional[Path]):
        """
        Initialize an instance of a TocGenerator.

        :param source_dir: The source directory where all html files reside.
        """

    def generate_toc_for(self, file: Path) -> Iterable[str]:
        """
        Generate a toctree directive for a given file.

        :param file: the file to generate the toctree directive for
        :return: a string interable representing the lines forming the toctree directive
        """
        return []


@dataclass
class _MenuEntry:
    title: str
    docname: str
    url: str
    children: List["_MenuEntry"]
    is_structural_dummy: bool = (
        False  # indicated whether a menu entry references a children's file as a structural dummy
    )
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        if not self.children:
            self.is_leaf = True
        else:
            self.is_leaf = False

    @staticmethod
    def from_json_node(json_node: Dict[str, Any]) -> "_MenuEntry":
        """Create a _MenuEntry from a json node (in doxygen's menudata.js).

        Note that this method will build up a _MenuEntry-tree automatically/recursively

        :param json_node: The json node to generate a _MenuEntry from
        :return: A _MenuEntry representation of the json_node and its' children
        """
        title = json_node["text"]
        url = json_node["url"]
        file = _MenuEntry._docname_from_url(url)
        children = _MenuEntry._get_sphinx_toc_compatible_children(json_node)
        is_structural_dummy = "is_structural_dummy" in json_node and json_node["is_structural_dummy"]
        return _MenuEntry(title, file, url, children, is_structural_dummy)

    @staticmethod
    def _docname_from_url(url: str) -> str:
        return url.split("#")[0].replace(".html", "")

    @staticmethod
    def _get_sphinx_toc_compatible_children(json_node: Dict[str, Any]) -> List["_MenuEntry"]:
        """Get a "sphinx compatible" view of the children.

        We therefore need a special handling for index anchors
        with doxygen we sometimes have urls in menu entries like:
        # - title: url
          - a: globals_enum.html#index_a
          - c: globals_enum.html#index_c
          - e: globals_enum.html#index_e
          - f: globals_enum.html#index_f
            ...
        The problem here is that the sphinx toctree simple cannot handle anchors... so we cannot add these
        links for entries in the parent's toctree. We therefore need to
        - eliminate all childrens with the same name/file down to one last child
        - then check if the parent has the same name/file and in that case get rid of the child completely
        """
        # get all children
        children = [_MenuEntry.from_json_node(c) for c in json_node["children"]] if "children" in json_node else []
        if not children:
            return []

        # get unique (considering .file value) children
        unique_children = []
        current_docname = _MenuEntry._docname_from_url(json_node["url"])
        unique_files = set()

        for child in children:
            if child.docname in unique_files and child.is_leaf:
                continue

            if child.docname == current_docname:
                json_node["is_structural_dummy"] = True

            unique_children.append(child)
            unique_files.add(child.docname)

        # if there is only one child item left and if that's the same as the current item - get rid of it
        current_docname = _MenuEntry._docname_from_url(json_node["url"])
        if len(unique_children) == 1 and unique_children[0].docname == current_docname and unique_children[0].is_leaf:
            json_node["is_structural_dummy"] = False
            return []

        return unique_children


class _CompoundKind(Enum):
    """\"Kind\" value in tag-file."""

    CATEGORY = auto()
    CLASS = auto()
    CONCEPT = auto()
    DIR = auto()
    ENUM = auto()
    EXCEPTION = auto()
    FILE = auto()
    GROUP = auto()
    INDEX_PAGE = auto()  # The index page, this doesn't appear in xml directly but needs special handling
    INTERFACE = auto()
    MODULE = auto()
    NAMESPACE = auto()
    PAGE = auto()
    PROTOCOL = auto()
    SERVICE = auto()
    SINGLETON = auto()
    STRUCT = auto()
    TYPE = auto()
    UNION = auto()
    UNKNOWN = auto()

    @classmethod
    def from_str(cls, kind_str: str) -> "_CompoundKind":
        """Get the kind corresponding to the str kind_str.

        returns _CompoundKind.UNKNOWN if not defined
        """
        kind = kind_str.upper()
        try:
            return cls[kind]
        except KeyError:
            _logger.warning(f"Unknown compound type {kind_str}")
            return cls.UNKNOWN


@dataclass
class _Compound:
    """Representation of <compound> from the tagfile."""

    class Ref(NamedTuple):
        """Reference to a _Compound, uniquely identifying it."""

        kind: _CompoundKind
        name: str

    kind: _CompoundKind
    name: str
    title: str
    filename: str
    children: List["_Compound.Ref"] = field(default_factory=list)

    CLASS_ALIASES: ClassVar[List[_CompoundKind]] = [
        _CompoundKind.CATEGORY,
        _CompoundKind.CLASS,
        _CompoundKind.ENUM,
        _CompoundKind.EXCEPTION,
        _CompoundKind.INTERFACE,
        _CompoundKind.MODULE,
        _CompoundKind.PROTOCOL,
        _CompoundKind.SERVICE,
        _CompoundKind.SINGLETON,
        _CompoundKind.TYPE,
    ]

    @property
    def ref(self) -> "_Compound.Ref":
        """The Reference to this instance."""
        return _Compound.Ref(self.kind, self.name)

    @classmethod
    def from_xml(cls, compound: XMLElement) -> "_Compound":
        """Parse a <compound> from XML."""
        kind_attr = compound.get("kind")
        if kind_attr is None:
            raise Exception("Couldn't parse tagfile, expected \"kind\" attribute wasn't found")

        kind = _CompoundKind.from_str(kind_attr)
        values: Dict[str, str] = {}
        required_keys: Iterable[str]

        if kind == _CompoundKind.GROUP:
            required_keys = ("name", "title", "filename")
        else:
            required_keys = ("name", "filename")

        for key in required_keys:
            if kind != _CompoundKind.GROUP and key == "title":
                continue

            node = compound.find(key)
            if node is None:
                raise Exception(f"Couldn't parse tagfile, expected sub-element {key}, but it wasn't found")
            if node.text is None:
                raise Exception(f"Couldn't parse tagfile, expected {key} to have a value, but it's empty")
            text: str = node.text
            values[key] = text

        title = values["title"] if "title" in values else values["name"]
        # We need to special case the index page, it shouldn't appear where other pages appear,
        # because it is already used as the root of the TOC hierarchy
        if kind == _CompoundKind.PAGE and values["name"] == "index":
            kind = _CompoundKind.INDEX_PAGE

        return cls(
            kind=kind,
            name=values["name"],
            title=title,
            filename=values["filename"],
            children=cls._get_children(compound, kind),
        )

    _CHILD_NODE_NAMES: ClassVar[Dict[_CompoundKind, str]] = {
        _CompoundKind.CLASS: "class",
        _CompoundKind.DIR: "dir",
        _CompoundKind.GROUP: "subgroup",
        _CompoundKind.NAMESPACE: "namespace",
        _CompoundKind.PAGE: "subpage",
    }

    @classmethod
    def _child_tag_for(cls, kind: _CompoundKind) -> Optional[str]:
        if kind in cls.CLASS_ALIASES:
            return cls._CHILD_NODE_NAMES[_CompoundKind.CLASS]

        if kind == _CompoundKind.INDEX_PAGE:
            return cls._CHILD_NODE_NAMES[_CompoundKind.PAGE]

        return cls._CHILD_NODE_NAMES.get(kind)

    @classmethod
    def _get_children(cls, compound: XMLElement, kind: _CompoundKind) -> List["_Compound.Ref"]:
        child_tag = cls._child_tag_for(kind)
        if child_tag is None:
            return []

        children: List["_Compound.Ref"] = []
        for subnode in compound.iterfind(f"./{child_tag}"):
            child_kind = kind  # Children are by default the same kind as the parent if not specified otherwise
            kind_attr = subnode.get("kind")
            if kind_attr is not None:
                child_kind = _CompoundKind.from_str(kind_attr)

            if subnode.text is None:
                raise Exception(f"Couldn't parse tagfile, expected {child_tag} tag to have a value, but its empty")
            child_name = subnode.text
            # Sub-pages are declared with the (.html) file extension for some reason
            if child_tag == "subpage":
                child_name = child_name.rsplit(".", maxsplit=1)[0]
            children.append(cls.Ref(child_kind, child_name))

        children.sort(key=operator.attrgetter("name"))
        return children


class _DoxygenTagFileCompounds:
    """Compounds from the tagfile."""

    def __init__(self, tagfile: Path) -> None:
        """Parse the tagfile and create the list of compounds from it."""
        self._compounds_by_name: Dict[_CompoundKind, Dict[str, _Compound]] = {}
        self._doxygen_version: Optional[Version]
        self._compounds_with_parents_cache: Optional[Set[_Compound.Ref]] = None
        self._parse(tagfile)

    @property
    def doxygen_version(self) -> Optional[Version]:
        """Version of doxygen that wrote the tagfile, or None if no version was written."""
        return self._doxygen_version

    def compounds_by_name(self, kind: _CompoundKind) -> Dict[str, _Compound]:
        """Get a dictionary with all compounds of kind, with the compound names as keys."""
        return self._compounds_by_name[kind]

    def __getitem__(self, key: _Compound.Ref) -> _Compound:
        """Get a compound based on its reference."""
        return self._compounds_by_name[key.kind][key.name]

    @property
    def kinds(self) -> Iterable[_CompoundKind]:
        """The compound kinds found in the tagfile."""
        return self._compounds_by_name.keys()

    def roots(self, kind: _CompoundKind) -> Iterable[_Compound.Ref]:
        """Get the compounds of kind with no parents, i.e. the roots of the hierarchy for kind."""
        compounds = {_Compound.Ref(kind, name) for name in self._compounds_by_name[kind]}
        return compounds - self._compounds_with_parents

    def _parse(self, tagfile: Path) -> None:
        root = defusedxml.ElementTree.parse(tagfile).getroot()
        if root is None:
            raise Exception('Couldn\'t parse tagfile "tagfile" tag not found')

        self._doxygen_version = self._parse_doxygen_version(root)

        for node in root.iterfind("./compound"):
            compound = _Compound.from_xml(node)
            self._compounds_by_name.setdefault(compound.kind, {})[compound.name] = compound

        def sorted_dict(d: Dict[str, _Compound]) -> Dict[str, _Compound]:
            return dict(sorted(d.items(), key=operator.itemgetter(0)))

        self._compounds_by_name = {k: sorted_dict(v) for k, v in self._compounds_by_name.items()}

    @staticmethod
    def _parse_doxygen_version(root: XMLElement) -> Optional[Version]:
        version_attrib = root.get("doxygen_version")
        if version_attrib is not None:
            return Version(version_attrib)

        # Earlier versions have a separate <doxygen> tag with a "version" attrobute
        doxygen_tag = root.find("./doxygen")
        if not doxygen_tag:
            return None

        version_attrib = doxygen_tag.get("version")
        if version_attrib is not None:
            return Version(version_attrib)

        return None

    @property
    def _compounds_with_parents(self) -> Set[_Compound.Ref]:
        if self._compounds_with_parents_cache is not None:
            return self._compounds_with_parents_cache

        all_compounds = chain.from_iterable(self._compounds_by_name[k].values() for k in self._compounds_by_name)
        self._compounds_with_parents_cache = set(chain.from_iterable(c.children for c in all_compounds))

        return self._compounds_with_parents_cache


class _CompoundConverter:
    """Converter from the XML format to _MenuEntries."""

    def __init__(self, compounds: _DoxygenTagFileCompounds):
        """Create a converter."""
        self.compounds = compounds
        self._menu_entries: Dict[_Compound.Ref, _MenuEntry] = {}

    def _get_or_create_entry_for(self, ref: _Compound.Ref) -> _MenuEntry:
        if ref in self._menu_entries:
            return self._menu_entries[ref]

        compound = self.compounds[ref]
        entry = _MenuEntry(
            title=compound.title,
            docname=_MenuEntry._docname_from_url(compound.filename),
            url=compound.filename,
            children=[self._get_or_create_entry_for(i) for i in compound.children],
        )
        self._menu_entries[compound.ref] = entry
        return entry

    def roots(self, kind: _CompoundKind) -> Iterable[_MenuEntry]:
        """Get an iterator to the root entries for the compounds of kind."""
        # Adds all groups to self._menu_entries as a side-effect, because all entries are reachable from the roots
        return (self._get_or_create_entry_for(r) for r in self.compounds.roots(kind))


class DoxygenTocGenerator:
    """
    A TocGenerator for doxygen.

    Will read the menudata.js to check whether a toctree
    directive needs to be generated or not.
    """

    def __init__(self, source_dir: Path, tagfile: Optional[Path]):
        """
        Initialize an instance of a TocGenerator.

        :param source_dir: The source directory where the doxygen html files reside.
        """
        self._source_dir = source_dir

        self._menu: _MenuEntry = self._load_menu_tree(source_dir / "menudata.js")

        # self._project_name, self._project_number = self._parse_project_infos()
        self._doxy_html_template: Tuple[str, str] = self._parse_template()

        # create rst files for those structural dummies doxygen is using...
        structural_dummies = [e for e in self._flatten_tree(self._menu) if e.is_structural_dummy]
        apply(structural_dummies, self._prepare_structural_dummy)
        apply(structural_dummies, self._create_toc_file_for_structural_dummy)

        self._menu_lookup: Dict[str, _MenuEntry] = self._create_menu_lookup(tagfile)

    _MENU_FILENAME_FOR_KIND: ClassVar[Dict[_CompoundKind, str]] = {
        _CompoundKind.PAGE: "pages",
        _CompoundKind.MODULE: "modules",
        _CompoundKind.NAMESPACE: "namespaces",
        # Disabled currently, because it would need more structure (e.g. grouping by namespace)
        # _CompoundKind.CONCEPT: "concepts",
        # _CompoundKind.CLASS: "annotated",
        # _CompoundKind.INTERFACE: "annotatedinterfaces",
        # _CompoundKind.STRUCT: "annotatedstructs",
        # _CompoundKind.EXCEPTION: "annotatedexceptions",
        # Similarly to classes, more structure would be required to present nicely.
        # _CompoundKind.FILE: "files",
    }

    @classmethod
    def _menu_filename_for(cls, kind: _CompoundKind, compounds: _DoxygenTagFileCompounds) -> Optional[str]:
        if kind == _CompoundKind.GROUP:
            if compounds.doxygen_version is not None and compounds.doxygen_version >= Version("1.9.8"):
                return "topics"
            return "modules"

        return cls._MENU_FILENAME_FOR_KIND.get(kind)

    def _extend_with_kind(
        self, kind: _CompoundKind, *, menu_lookup: Dict[str, _MenuEntry], converter: _CompoundConverter
    ) -> None:
        menu_parent = self._menu_filename_for(kind, converter.compounds)
        _logger.debug(f"menu_parent: {menu_parent}")
        if menu_parent is None:
            return

        if menu_parent not in menu_lookup:
            _logger.debug("skipping")
            return

        # Flatten and merge the new keys into menu_lookup
        menu_lookup.update(
            (e.docname, e) for e in chain.from_iterable((self._flatten_tree(root) for root in converter.roots(kind)))
        )
        children: List[_MenuEntry] = menu_lookup[menu_parent].children
        children.extend(converter.roots(kind))

        menu_lookup[menu_parent] = replace(menu_lookup[menu_parent], children=children)

    def _extend_menu_with_tagfile(self, menu_lookup: Dict[str, _MenuEntry], tagfile: Optional[Path]) -> None:
        compounds = self._parse_tagfile(tagfile)
        if compounds is None:
            return

        converter = _CompoundConverter(compounds)
        for kind in compounds.kinds:
            _logger.debug(f"Extending with kind: {kind}")
            self._extend_with_kind(kind, menu_lookup=menu_lookup, converter=converter)

    def _create_menu_lookup(self, tagfile: Optional[Path]) -> Dict[str, _MenuEntry]:
        menu_lookup = {e.docname: e for e in self._flatten_tree(self._menu)}
        self._extend_menu_with_tagfile(menu_lookup, tagfile)
        return menu_lookup

    @staticmethod
    def _parse_tagfile(tagfile: Optional[Path]) -> Optional[_DoxygenTagFileCompounds]:
        if tagfile is None:
            _logger.info("Skipping toc generation, tagfile not specified")
            return None

        try:
            parser = _DoxygenTagFileCompounds(tagfile)
            return parser
        except FileNotFoundError as err:
            _logger.warning(f"Failed to parse tagfile: {err}")
            _logger.info("Skipping toc generation")
            return None

    def _parse_template(self) -> Tuple[str, str]:
        """Parse a "doxygen html template shell" out of the index.html file.

        :return: A Tuple containing the doxygen html before the content area and the content after the content area.
        """
        # load html file as string and remove the newline chars
        blueprint = self._source_dir / "index.html"
        complete_html = blueprint.read_text()
        linearized_html = complete_html.replace("\n", "").replace("\r", "")

        # split the html string on the content element
        # (so that we can use the 2 parts and inject our content in the middle)
        split_start_search = r"<!--header--><div class=\"contents\">"
        split_end_search = r"</div><!-- contents -->"
        split_regex = re.compile(f"{split_start_search}.*?{split_end_search}")
        splitted = split_regex.split(linearized_html)
        if len(splitted) != 2:
            raise Exception(
                "couldn't parse html template for toc dummies from index.html. "
                "Maybe the format of doxygen has changed? Or do you have a custom template? In that case: "
                "we search for the following regex to find anything except the content: "
                '<!--header--><div class="contents">.*</div><!-- contents -->'
            )
        prefix, suffix = splitted

        # replace the original index title with a marker that we can easily replace afterwards
        replace_regex = re.compile('(?<=<div class="title">).*?(?=</div>)')
        prefix_replaced = replace_regex.sub("@@@-TITLE-@@@", prefix)

        return prefix_replaced + split_start_search.replace('\\"', '"'), split_end_search + suffix

    def _sanitize_filename(self, value: str) -> str:
        """Sanitize value to make it usable as a filename.

        - Try to replace unicode characters with ascii fallbacks
        - drop any remaining non-ascii characters
        - converts to lower case
        - replace whitespace and slashes with underscores
        - keeps only alphanumerics, dash and underscore
        """
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
        value = re.sub(r"[\s/]", "_", value.lower())
        return re.sub(r"[^\w\-_]", "", value)

    def _prepare_structural_dummy(self, structural_dummy: _MenuEntry):
        clean_title = self._sanitize_filename(structural_dummy.title)
        toc_docname = f"{structural_dummy.docname}_{clean_title}"
        structural_dummy.docname = toc_docname

    def _create_toc_file_for_structural_dummy(self, structural_dummy: _MenuEntry):
        prefix, suffix = self._doxy_html_template

        content = [
            f".. title:: {structural_dummy.title}",
            "",
            f"{structural_dummy.title}",
            f"{'-' * len(structural_dummy.title)}",
            "",
            ".. container:: doxygen-content",
            "",
            "   .. raw:: html",
            "",
            "      " + prefix.replace("@@@-TITLE-@@@", structural_dummy.title),
            "",
            "   .. toctree::",
            "      :maxdepth: 4",
            "",
            *[f"      {item.title} <{item.docname}>" for item in structural_dummy.children],
            "",
            "   .. raw:: html",
            "",
            "      " + suffix,
            "",
        ]

        file = self._source_dir / f"{structural_dummy.docname}.rst"

        write_file(file, content)

    def _load_menu_tree(self, menu_data_js_path: Path) -> _MenuEntry:
        menu = read_js_data_file(menu_data_js_path)["menudata"]
        items = menu["children"]

        children = [_MenuEntry.from_json_node(c) for c in items]
        root, *_ = children
        _, *children_without_root = children
        root_copy = replace(root, children=children_without_root)
        return root_copy

    def _flatten_tree(self, *entries: _MenuEntry) -> Iterator[_MenuEntry]:
        for entry in entries:
            yield entry
            if not entry.is_leaf:
                yield from self._flatten_tree(*entry.children)

    def generate_toc_for(self, file: Path) -> Iterator[str]:
        """
        Generate a toctree directive for a given file.

        Note that the toctree will only be generated when the file is part of a menu
        structure.
        :param file: the file to generate the toctree directive for
        :return: a string iterator representing the lines forming the toctree directive
        """
        name = file.stem
        if name in self._menu_lookup:
            _logger.debug(f"Generating toc for {name}")
            matching_menu_entry = self._menu_lookup[name]

            children = matching_menu_entry.children
            if not children:  # when the children list is empty no tocs need to be generated.
                return

            yield ".. toctree::"
            yield f"   :caption: {matching_menu_entry.title}"
            yield "   :maxdepth: 2"
            yield "   :hidden:"
            yield ""
            yield from [f"   {item.title} <{item.docname}>" for item in matching_menu_entry.children]
            yield ""
