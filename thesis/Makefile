SRCDIR = src
OUTDIR = out
SRC = $(sort $(wildcard $(SRCDIR)/*.md))
HEADER = $(SRCDIR)/header.tex
TITLEPAGE = $(SRCDIR)/frontmatter.tex
BIB = $(SRCDIR)/bibliography.bib
METADATA = $(SRCDIR)/00-meta.yaml
FILENAME = gentiletti_emanuele_thesis_24-25

FLAGS = -t latex --toc --pdf-engine=lualatex --bibliography=$(BIB)
FLAGS += --top-level-division=chapter --resource-path=src
FLAGS += --table-caption-position=below
FLAGS += -F pandoc-crossref --citeproc

$(OUTDIR)/$(FILENAME).pdf $(OUTDIR)/$(FILENAME).tex: $(SRC) $(HEADER) $(TITLEPAGE) $(METADATA)
	pandoc $(FLAGS) -B $(TITLEPAGE) $(METADATA) $(SRC) -o $@ --csl apa.csl

