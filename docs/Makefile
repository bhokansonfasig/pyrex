# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = PyREx
SOURCEDIR     = .
BUILDDIR      = build

PDF = ../PyREx Documentation.pdf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Clean should remove all numpy docstring helpers and any cached items
clean:
	-rm -r $(BUILDDIR)/*
	-rm docstrings/*
	-rm api/docstrings/*

# Customize html command to place .nojekyll file in build/html directory
html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	touch $(BUILDDIR)/html/.nojekyll

# Customize latexpdf command to place pdf outside of build directory
latexpdf:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp $(BUILDDIR)/latex/*.pdf "$(PDF)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)