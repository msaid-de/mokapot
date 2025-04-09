# Handling the limits on maximum open files

## Why and how to use `ulimit`

`brew_rollup` may be used with hundreds or even thousands of files to be merged together.
For security reasons, the operating system usually puts a limit on how many files can be opened simultaneously by one
process.
On POSIX systems like Linux there is the `ulimit` command to control limits such as this.
For `ulimit` to show or modify the limit on open files there is the `-n` switch.
Furthermore, for each of the limits (there is also limits for memory usage, cpu usage, pipes, etc.) there is soft and a
hard limit.
The soft limit is the limit that the (child) processes spawned from the shell see -
the hard limit is always larger or equal to the soft limit and is the absolute maximum to which the soft limit can be
raised.
You can instruct `ulimit` to set or show the soft limit by including the `-S` switch, and the hard limit by including
the `-H` switch.
The default when showing limits is to show the soft limit, while the default when setting limits is to set the soft
*and* the hard limit simultaneously.
Unfortunately, the default for setting limits is completely braindead IMHO, as we'll see in a second.
When I issue `ulimit -n` or `ulimit -Sn` on my system, I get 1024 as the soft, and with `ulimit -Hn` I get 1048576 as
the hard limit.
When I set now a new limit with say `ulimit -n 4000` I raise the soft limit to 4000, but at the same time also *lower*
the hard limit to 4000.
So, if I then realize that 4000 wasn't enough and try `ulimit -n 6000`, this fails with "cannot modify limit: Operation
not permitted", because the hard limit is set already to 4000 and you can neither set the soft limit higher than the
hard limit nor increase the hard limit.
(NB you can raise the hard limit if you have root privileges, but since you usually don't have that in a shell and you
can't do that within a shell (only when spawning a new shell) this is practically impossible.)
So, the moral of the story is: *always* use `ulimit -Sn xyz` to set the (soft) limit and never `ulimit -n xyz` like many
sources on the internet will tell you.

## Possible alternatives

Note: there is also the possibility to set those limits in `/etc/security/limits.conf` as defaults for every new shell.
E.g.

```
   msaid    hard    nofile    65536
   msaid    soft    nofile    16384
```

would set the hard limit for user `msaid` to 65536 and the soft limit to 16384.
However, I would recommend leaving the relatively low soft limit as is and only raising it in a shell when needed, as
this can protect from bugs and run-away processes consuming system resources unnecessarily.

IMHO maybe the best way to go would be to wrap `brew_rollup` in a script, count the number of files in the directory
containing all the input files, add a small number like 10 on top of it, set the limit accordingly and then run
`brew_rollup`.

Another way would be to use `prlimit`, which can also be used with a command: e.g.
`prlimit -n=20000 python -m mokapot.brew_rollup ...`.
This only sets the limits for the spawned process and not for the current shell.
And of course instead of the plain 20000 you can use something like `$max_open` with
`max_open=$(($(ls $datadir | wc -l)+10))`.

Example

```
#!/bin/bash

# Get size of datadir (doesn't matter if we have some additional files in there)
datadir="$1"
max_open=$(($(ls $datadir | wc -l)+10))

# and then either
prlimit -n=$max_open python -m mokapot.brew_rollup -s $datadir [options]

# or (here the -S doesn't really matter, it's only for the runtime of the script anyway)
ulimit -Sn $max_open
python -m mokapot.brew_rollup -s $datadir [options]
```
