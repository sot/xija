core.so: core.c
	gcc -fPIC -O3 -c -Wall core.c
	ld -shared -soname libcore.so -o libcore.so -lc core.o
