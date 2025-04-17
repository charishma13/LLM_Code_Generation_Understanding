# https://github.com/docling-project/docling
from docling.document_converter import DocumentConverter

source = "https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-non-cuboid-cell-10a-cutoff"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"