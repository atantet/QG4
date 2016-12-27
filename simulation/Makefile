INCDIR=-I$(HOME)/local/include/
LIBDIR=-L$(HOME)/local/lib/
LIBS=-lgsl -lgslcblas -lconfig++ -lergopack -lgomp
CONFIG=../cfg/readConfig

CC=g++
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -O3
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

<<<<<<< HEAD
#EXE=simQG4
EXE=simQG4Sprinkle
=======
# Use OpenMP?
ifeq ($(WITH_OMP),1)
  LIBS +=-lgomp
  CFLAGS += -fopenmp -DWITH_OMP=$(WITH_OMP)
endif

EXE=simQG4
#EXE=simQG4Sprinkle
>>>>>>> f22b88b3bd8cb8f5e887873e50668ba85b86cb61

all:$(EXE).out
	mkdir -p ../results/simulation/

$(EXE).o:$(EXE).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(EXE).out
