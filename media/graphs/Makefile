
SOURCES := $(wildcard *.gv)
IMAGES := $(SOURCES:.gv=.png)

all: $(IMAGES)

%.dot.png: %.dot.gv
	dot -Tpng -o $@ $<