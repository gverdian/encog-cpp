IDIR =./include
CC=gcc
CFLAGS=-I$(IDIR) -std=c99 -pedantic -Wall

ODIR=obj
LDIR =./lib

LIBS=-lm

_DEPS = encog.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = encog.o test.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

encog: $(OBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
