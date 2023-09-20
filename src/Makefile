include ../shared/common.mk
SHARED=../shared
DEP= $(SHARED)/timer.o $(SHARED)/graph.o $(SHARED)/argument_parsing.o

all: tc-sync.o tc-sync

tc-sync.o: tc-sync.cu 
	$(NC) -c tc-sync.cu  $(CFLAGS) $(NFLAGS)

tc-sync: tc-sync.o $(DEP)
	$(NC) tc-sync.o $(DEP) -o tc-sync $(CFLAGS) $(NFLAGS)
	rm -f tc-sync.o

clean: 
	rm -f tc-sync