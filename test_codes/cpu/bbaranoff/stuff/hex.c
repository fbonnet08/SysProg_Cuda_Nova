/*
 * Read hex strings and output as text.
 *
 * No checking of the characters is done, but the strings must have an even
 * length.
 *
 * $Id: hex2ascii.c,v 1.1 2009/09/19 23:56:49 grog Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "suppress_space.h"
char hexdigit (char c)
{
  char outc;

  outc = c -'0';
  if (outc > 9)                                 /* A - F or a - f */
    outc -= 7;                                  /* A - F */
  if (outc > 15)                                /* a - f? */
    outc -= 32;
  if ((outc > 15) || (outc < 0))
  {
    fprintf (stderr, "Invalid character %c, aborting\n", c);
    exit (1);
  }
  return outc;
}
char test[]="31 32 33 34 35 36 37 38";
int main()
{  int arg;
  char *c=spaces(test);
  int sl;
  char oc;

  for (arg = 0; arg < 3; arg++)
  {
    sl = strlen (c);
    if (sl & 1)                                 /* odd length */
    {
      fprintf (stderr,
               "%s is %d chars long, must be even\n",
               c,
               sl );
      return 1;
    }
    while (*c)
    {
      oc = (hexdigit (*c++) << 4) + hexdigit (*c++);
      fputc (oc, stdout);
    }
  }
return 1;}



