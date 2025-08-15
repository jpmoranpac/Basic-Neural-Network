PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SRCDIR  	 := $(PROJECT_ROOT)/src/
OBJDIR  	 := $(PROJECT_ROOT)/obj/
BINDIR  	 := $(PROJECT_ROOT)/bin/
TARGET    	 := neural_network.out
EXE     	 := $(BINDIR)$(TARGET)

SFILES  	 := cpp
OFILES  	 := o
CC      	 := g++
INCFLAGS 	 := -I$(PROJECT_ROOT)
CPPFLAGS 	 := -g $(INCFLAGS)
	 
SRCS 	     := $(shell find $(SRCDIR) -name "*.$(SFILES)")
OBJS     	 := $(patsubst $(SRCDIR)%.$(SFILES), $(OBJDIR)%.$(OFILES), $(SRCS))

.PHONY: default all clean

default: $(EXE)

all: clean default

folders:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)

$(EXE): $(OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@

$(OBJDIR)%$(OFILES): $(SRCDIR)%$(SFILES) | folders
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJS) $(EXE)
	@rmdir $(OBJDIR)
	@rmdir $(BINDIR)