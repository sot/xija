core6.so: core6.c
	gcc -fPIC -O3 -c -Wall core6.c
	ld -shared -soname libcore.so.6 -o libcore.so.6 -lc core6.o
	rm -f libcore.so
	ln -s libcore.so.6 libcore.so
