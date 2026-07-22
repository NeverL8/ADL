all: build/transformer.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/thesis.pdf: FORCE | build
	latexmk $(TeXOptions) transformer.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
