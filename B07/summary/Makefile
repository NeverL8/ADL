all: build/MNIST_diffusion.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/thesis.pdf: FORCE | build
	latexmk $(TeXOptions) MNIST_diffusion.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
