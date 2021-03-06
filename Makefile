# Name of your program:
NAME=RNN

# List of all .cpp source code files included in your program (separated by spaces):
SRC=core.cpp serialized/matrix.cpp serialized/vect.cpp io.cpp main.cpp



SRCPATH=./src/
OBJ=$(addprefix $(SRCPATH), $(SRC:.cpp=.o))

RM=rm -f
INCPATH=./include
CPPFLAGS+= -std=c++0x -I $(INCPATH) -g


all: $(OBJ)
	g++  $(OBJ) -o $(NAME)

clean:
	-$(RM) *~
	-$(RM) *#*
	-$(RM) *swp
	-$(RM) *.core
	-$(RM) *.stackdump
	-$(RM) $(SRCPATH)*.o
	-$(RM) $(SRCPATH)*.obj
	-$(RM) $(SRCPATH)*~
	-$(RM) $(SRCPATH)*#*
	-$(RM) $(SRCPATH)*swp
	-$(RM) $(SRCPATH)*.core
	-$(RM) $(SRCPATH)*.stackdump

fclean: clean
	-$(RM) $(NAME)

re: fclean all
