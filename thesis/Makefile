SRCDIR = src
OUTDIR = out
SRC = $(sort $(wildcard $(SRCDIR)/*.md))
HEADER = $(SRCDIR)/header.tex
TITLEPAGE = $(SRCDIR)/frontmatter.tex
BIB = $(SRCDIR)/bibliography.bib
METADATA = $(SRCDIR)/00-meta.yaml
FILENAME = gentiletti_emanuele_thesis_24-25

FLAGS = -t latex --toc --pdf-engine=lualatex --citeproc --bibliography=$(BIB)
FLAGS += --top-level-division=chapter --resource-path=$(SRCDIR) 

$(OUTDIR)/$(FILENAME).pdf $(OUTDIR)/$(FILENAME).tex: $(SRC) $(HEADER) $(TITLEPAGE) $(METADATA)
	pandoc $(FLAGS) -H $(HEADER) -B $(TITLEPAGE) $(METADATA) $(SRC) -o $@ --csl apa.csl

