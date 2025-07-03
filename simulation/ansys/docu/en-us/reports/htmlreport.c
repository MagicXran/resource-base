/*   Function to work with HTML templates

     $Revision$
     $Date$

     Function rep_html.c     written by C.W.Aiken   6/2/98
     Function make_lower.c   written by C.W.Aiken   6/2/98
     Function process_html.c written by C.W.Aiken   7/15/98
     Function process_grph.c written by C.W.Aiken   7/30/98
     Function to_ansys_output      written by C.W.Aiken   6/11/98

     Function evl.c          written by Greg Sharpe 6/9/98
     Function getnexytjpg    written by Greg Sharpe 6/11/98
     Function query_jpg      written by Greg Sharpe 6/11/98

*/

#include<stdio.h>
#include<string.h>
#include<stdlib.h>

#define DEBUG 0
#define DEBUG1 0

/* declare the functions that are used */
int rep_html(char*);
void make_lower(char*,char*);
void process_html(char*,char*,char*);
void process_grph(int*,char*,char*,char*,char*,char*,char*);
void to_ansys_output(char*);
void send_to_ansys(char*);
void send_to_scratch(char*);

/* functions from Greg Sharpe */
int evl_get(char*);
int getnextjpg(char*);
void query_jpg(char*);

/* declare functions that are not in this file */
extern int cAnsSendCommand(char*);  
extern int cAns_printff(const char *format, ...);
extern void share_error(int, void*);
extern int cAnsGetValue(char*,double*,char*,int*);

#ifdef WIN32
#  define TYPECALL __stdcall
#  define psfnam_ PSFNAM
#  define htmlgui_ HTMLGUI
#else
#  if defined (_AIX) || defined (HP700_SYS)
#    define TYPECALL
#    define psfnam_ psfnam
#    define htmlgui_ htmlgui
#  else
#    define TYPECALL
#  endif
#endif
extern void TYPECALL psfnam_( int *, int *, int *);
extern int TYPECALL htmlgui_( int *, int *);  

/* set up globals */
FILE *fp_scratch;
int  scratch_open=0;
char ansys_tag_start[] ="<ansys>";
char ansys_tag_stop[]  ="</ansys>";
char ansys_tag_start4[]="&lt;ansys>";     /* Netscape 4 composer tag */
char ansys_tag_stop4[] ="&lt;/ansys>";    /* Netscape 4 composer tag */
char ansys_tag_gstart[] ="<ansysgr";
char ansys_tag_gstop[]  ="</ansysgr>";
char ansys_tag_gstart4[]="&lt;ansysgr";   /* Netscape 4 composer tag */
char ansys_tag_gstop4[] ="&lt;/ansysgr>"; /* Netscape 4 composer tag */

int rep_html(char* ansys_line)
{
   /*
      PURPOSE:
      Read a html template file and form a report for ansys.
      Send any information between <ansys> & </ansys> tags
      to ansys to run and have the results inserted into the
      html report. Send any information between <ansysgr> & </ansysgr>
      tags to ansys to form a plot and have the plot inserted
      into the html report as a jpeg image.

      INPUT:
      char* ansys_line   ansys command line in form of
                         ~html,report_name,template_name
                         report_name defaults to "report.html"
                         template_name defaults to "template.html"
   */

   FILE *fp_template, *fp_scratch;
   void* zip = 0;
   char line_in[256], line_lc[256], line_work[256], jpeg_name[256];
   char path[256];

   int i,j,k,graph,netscape,report_number=0;
   char *j1, *j2, *j3;
   char* ptr_c;
   char *squote_ptr;

   char template_file[256];
   char report_name[256];
   char report_dir[256];
   
/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */
   if(DEBUG1) printf("\n\nDEBUG: enter rep_html\n");
   j1=&ansys_line[0];
   j2='\0';
   j3='\0';

   j2=strstr(j1,",");
   if(j2) 
   {
      j2++;
      j3=strstr(j2,",");
      if(j3)
         j3++;
      else
         j3='\0';
   }
   else
      j2='\0';

   /* get report name */
   if(j2)
   {
      strtok(j2,",");
      strcpy(report_name,j2);
   }
   else
   {
      j2=&report_name[0];
      strcpy(j2,"report.html");
   }

   /* get template name */
   if(j3)
   {
      squote_ptr = NULL;
      strtok(j3,",");
      strcpy(template_file,j3);
      /* strip off single quotes */
      squote_ptr = strstr(template_file,"'");
      if (squote_ptr)
      {
         squote_ptr++;
         strcpy(template_file,squote_ptr);
         squote_ptr = strstr(template_file,"'");
         if (squote_ptr) *squote_ptr = '\0';
      }
   }
   else
   {
      j3=&template_file[0];
      strcpy(j3,"template.html");
   }

   /* if the report name does NOT have an HTML extension, add one */
   ptr_c=strstr(report_name,".html");
   if(! ptr_c)
      strcat(report_name,".html");

   /* set up the report directory */
   strcpy(report_dir,report_name);
   ptr_c=strstr(report_dir,".");
   *ptr_c='\0';
   
   if(DEBUG1) printf("DEBUG: Report name: %s\n",report_name);
   if(DEBUG1) printf("DEBUG: Report dir: %s\n",report_dir);
   if(DEBUG1) printf("DEBUG: Template name: %s\n",template_file);

   /* create a directory for final report data */
#ifdef WIN32
   strcpy(line_work,"~tcl,'file delete -force ");
   strcat(line_work,report_dir);
   strcat(line_work,"'");
   if(DEBUG1) printf("DEBUG: Delete: %s\n",line_work);
   send_to_ansys(line_work);
   strcpy(line_work,"~tcl,'file mkdir ");
   strcat(line_work,report_dir);
   strcat(line_work,"'");
   if(DEBUG1) printf("DEBUG: Create: %s\n",line_work);
   send_to_ansys(line_work);
#else
   strcpy(line_work,"rm -rf ");
   strcat(line_work,report_dir);
   if(DEBUG1) printf("DEBUG: Delete: %s\n",line_work);
   system(line_work);
   strcpy(line_work,"mkdir ");
   strcat(line_work,report_dir);
   if(DEBUG1) printf("DEBUG: Create: %s\n",line_work);
   system(line_work);
#endif
   
/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */

   /* try to open the template file */
   if( (fp_template=fopen(template_file,"rt")) == NULL)
   {
      printf("File: %s was NOT opened !\a\n", template_file);
      share_error(1002,zip);
      return(1);
   }

   /* Tell Ansys where the report directory is */
   send_to_ansys(")/nopr");
	strcpy(line_work,")*dim,_rdir,char,8");
	send_to_ansys(line_work);
	strcpy(line_work,")_rdir(1)='");
	j=0;
	k=strlen(line_work);
	for(i=0; i<(int)strlen(report_dir); i++)
	{
		if(j < 8)
		{
         line_work[k++]=report_dir[i];
			j++;
		}
		else
		{
			line_work[k++]='\'';
			line_work[k++]=',';
			line_work[k++]='\'';
         line_work[k++]=report_dir[i];
			j=0;
		}
	}
	line_work[k++]='\'';
	line_work[k]='\0';
	send_to_ansys(line_work);

   /* Tell Ansys to open the report file */
   strcpy(line_work,")/output,");
   strcat(line_work,report_name);
   send_to_ansys(line_work);

   /* the file was opened OK. Now try to read the file. */
   scratch_open=0;
   while ( (fgets(line_in,256,fp_template)) != NULL)
   {
      if(DEBUG1) printf("\n\nDEBUG: In: %s",line_in);

      /* move input line to a work space, lowercase work space */
      make_lower(line_in,line_lc);
      if(DEBUG) printf("DEBUG: Work: %s\n",line_lc);

/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */
      /* see if this is an <ANSYS> tag */
      netscape=0;
      ptr_c=strstr(line_lc,ansys_tag_start);
      if(! ptr_c)
      {
         ptr_c=strstr(line_lc,ansys_tag_start4);  /* Netscape 4 */
         if(ptr_c) netscape=1;
      }

      if(ptr_c)
      {
         graph=0;
         if(DEBUG) printf("DEBUG: start tag match: %s\n",line_lc);

         /* if ansys tag is not at beginning of line copy the
            data to left of ansys tag to the report file */
         if(ptr_c != line_lc)
         {
            i=ptr_c - line_lc;   /* number of chars to left of ansys tag */
            strcpy(line_work,line_in); /* use original line, not line_lc */
            line_work[i]='\0';
            to_ansys_output(line_work);
         }
         ptr_c=ptr_c + 7;      /* reset pointer to first char past ansys tag */
         if(netscape) ptr_c=ptr_c + 3;
         i=ptr_c - line_lc;    /* index to first char past ansys tag */ 
         ptr_c=&line_in[i];    /* shift original line to left */
         strcpy(line_in,ptr_c);
         ptr_c=&line_lc[i];    /* shift lower case line to left */
         strcpy(line_lc,ptr_c);
         if(DEBUG) printf("DEBUG: Call process_html: %s\n",line_in);
         process_html(line_in, line_lc, line_work);
         continue;
      }

/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */
      /* see if this is an <ANSYSGR> tag */
      netscape=0;
      ptr_c=strstr(line_lc,ansys_tag_gstart);
      if(! ptr_c)
      {
         ptr_c=strstr(line_lc,ansys_tag_gstart4);  /* Netscape 4 */
      }

      if(ptr_c)
      {
         graph=1;
         if(DEBUG) printf("DEBUG: graph start tag match: %s\n",line_lc);

         /* if ansys tag is not at beginning of line copy the
            data to left of ansys tag to the report file */
         if(ptr_c != line_lc)
         {
            i=ptr_c - line_lc;   /* number of chars to left of ansys tag */
            strcpy(line_work,line_in); /* use original line, not line_lc */
            line_work[i]='\0';
            to_ansys_output(line_work);
            ptr_c=&line_lc[i];
         }
      
         /*
         ptr_c=ptr_c + 9;     
         if(netscape) ptr_c=ptr_c + 3;
         */

         i=ptr_c - line_lc;    
         ptr_c=&line_in[i];    
         strcpy(line_in,ptr_c);
         ptr_c=&line_lc[i];   
         strcpy(line_lc,ptr_c);
         if (DEBUG) printf("DEBUG: Call process_grph: %s\n",line_in);
         process_grph(&report_number, line_in, line_lc, line_work, jpeg_name, report_dir, path);
         continue;
      }
/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */


      /* if send_to_ansys is active, process the line */
      if(scratch_open)
      {
         if (DEBUG) printf("DEBUG: b-process: %s\n",line_lc);
         if(graph)
            process_grph(&report_number, line_in, line_lc, line_work, jpeg_name, report_dir, path);
         else
            process_html(line_in, line_lc, line_work);
      }
      else
      {
         to_ansys_output(line_in);
      }
   }
   fclose(fp_template);
   send_to_ansys(")/go");
   send_to_ansys(")/out");
   send_to_ansys(")_rdir=");

/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */

   /* the report should be finished. Move report to the report directory */
#ifdef WIN32
   strcpy(line_work,".\\");
#else
   strcpy(line_work,"./");
#endif
   strcat(line_work,report_dir);
#ifdef WIN32
   strcat(line_work,"\\");
#else
   strcat(line_work,"/");
#endif
   strcat(line_work,report_name);
   rename(report_name,line_work);
   printf("\n\nGeneration of report %s is complete.\n",report_name);
   printf("All report information can be found in directory %s\n",report_dir);
   if(DEBUG1)   printf("DEBUG: leave rep_html\n");
   share_error(1002,zip);
   return(0);
}

/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */
/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */

/* lowercase work space */
void make_lower(char* line_in, char* line_work)
{
   /*
      PURPOSE:
      Copy an input line from the template file to
      a work string and lower case the work string.
      
      INPUT: 
      char *line_in     line from template file

      OUTPUT:
      char *line_work   lower case form of the input line
   */
   while (*line_in)
   {
      if(*line_in == 015)
         break;
      *line_work++ = tolower(*line_in++);
   }
   *line_work='\0';
   return;
}

/* an ansys start tag was found, process the line */
void process_html(char* ptr_in, char* ptr_lc, char* ptr_w)
{

   /*
      PURPOSE: 
      Send lines within the <ansys> & </ansys> tags to Ansys

      INPUT:
      char *ptr_in   original input line from template file
      char *ptr_lc   original line lower cased
      char *ptr_w    work string
   */

   int j,netscape=0;
   char* ip;
   char* ptr_c;

   /* if scratch_open is not active, then make active */
   ip=ptr_lc;
   if(scratch_open == 0)
   {
      send_to_scratch("/nopr\n");
   }

   /* move to work */
   strcpy(ptr_w,ip);
   if(DEBUG) printf("DEBUG: after strip: %s\n",ptr_w);

   /* see if there is an </ANSYS> tag on this line */
   netscape=0;
   ptr_c=strstr(ptr_w,ansys_tag_stop);
   if(! ptr_c)
   {
      ptr_c=strstr(ptr_w,ansys_tag_stop4);
      netscape=1;
   }
   if(ptr_c)
   {
      if(DEBUG) printf("DEBUG: stop match: %s\n",ptr_w);

      /* </ansys> tag is at start of line */
      if(ptr_c == ptr_w)
      {
         /* run the scratch file */
         scratch_open=2;
         send_to_scratch(" ");

         /* get any data after the </ansys> tag */
         j=ptr_c - ptr_w + 8;
         if(netscape) j=j + 3;
         ptr_c=&ptr_in[j]; /* use original line, not lower case */
         to_ansys_output(ptr_c);
         return;
      }

      /* </ansys> tag is at middle or end of line */
      else
      {
         j=ptr_c - ptr_w;
         strcpy(ptr_w,ptr_in); /* use original line, not lower case */
         ptr_w[j]='\n';
         ptr_w[j+1]='\0';
         send_to_scratch(ptr_w);

         /* run the scratch file */
         scratch_open=2;
         send_to_scratch(" ");

         /* get any data after the </ansys> tag */
         j=j + 8;
         if(netscape) j=j + 3;
         ptr_c=&ptr_in[j];
         strcpy(ptr_w,ptr_c); /* use original line, not lower case */
         to_ansys_output(ptr_w);
         return;
      }
   }
   
   /* send to ansys for processing */
   send_to_scratch(ptr_in); /* use original line, not lower case */

   return;
}

/* an ansysgr start tag was found, process the line */
void process_grph(int* report_number, char* ptr_in, char* ptr_lc, char* ptr_w, \
                  char* ptr_j, char* ptr_dir, char* ptr_path)
{

   /*
      PURPOSE:
      Send lines within the <ansysgr> & </ansysgr> tags to Ansys.
      The <ansysgr> tag may contain a jpeg name in the form of
      <ansysgr name=xxx.jpg>. After the ansys jpeg is formed
      this function will move the jpeg to the report directory
      and assign it the name from the <ansysgr> tag.

      INPUT:
      int  *report_number   an incremental number to be used when
                            <ansysgr> tags do not have a jpeg name.
                            the default name of "reportxx.jpg" (where
                            xx is the incremental number) will be used.

      char *ptr_in          original input line from the template file
      char *ptr_lc          original line lower cased
      char *ptr_w           work string
      char *ptr_j           string space for jpeg name
      char *ptr_dir         report directory
      char *ptr_path        work space to form complete path names
   */

   int j,netscape=0;
   char *ip;
   char *ptr_c;

   /* if scratch_open is not active, then make it active */
   ip=ptr_lc;
   if(scratch_open == 0)
   {
      send_to_scratch("/nopr\n");

      /* get JPG name for report from "name=" on the <ansysgr> tag */
      ip=strstr(ptr_lc,"name=");
      if(ip)
      {
         ip=ip+5;
         strcpy(ptr_path,ip);
         ip=strstr(ptr_path,">");
         *ip='\0';
      }
      else
      {
         *report_number=*report_number+1;
         sprintf(ptr_path,"report_grph%2.2d.jpg",*report_number);
         printf("\n\nJPEG name NOT FOUND on <ANSYSGR> tag.\n");
         printf("JPEG name of \"%s\" will be used.\n",ptr_path);
      }
      if(DEBUG) printf("DEBUG: JPEG name: %s\n",ptr_path);
      getnextjpg(ptr_j);
      if(DEBUG) printf("DEBUG: Next generated JPEG name: %s\n",ptr_j);
      ip=strstr(ptr_lc,">");
      ip++;
   }

   /* move to work */
   strcpy(ptr_w,ip);
   if(DEBUG) printf("DEBUG: after strip: %s\n",ptr_w);
	if( strlen(ptr_w) <= 1 ) 
		return;

   /* see if there is an </ANSYSGR> tag on this line */
   netscape=0;
   ptr_c=strstr(ptr_w,ansys_tag_gstop);
   if(! ptr_c)
   {
      ptr_c=strstr(ptr_w,ansys_tag_gstop4);
      netscape=1;
   }
   if(ptr_c)
   {
      if(DEBUG) printf("DEBUG: graph stop match: %s\n",ptr_w);

      /* </ansysgr> tag is at start of line */
      if(ptr_c == ptr_w)
      {
         /* run the scratch file */
         scratch_open=2;
         send_to_scratch(" ");

         strcpy(ptr_w,"<img src=\"");
         strcat(ptr_w,ptr_path);
         strcat(ptr_w,"\" alt=\"\" width=483 height=363>");
         if(DEBUG) printf("DEBUG: IMG: %s\n",ptr_w);
         to_ansys_output(ptr_w);

         /* move jpeg file to report directory */
#ifdef WIN32
         strcpy(ptr_w,".\\");
#else
         strcpy(ptr_w,"./");
#endif
         strcat(ptr_w,ptr_dir);
#ifdef WIN32
         strcat(ptr_w,"\\");
#else
         strcat(ptr_w,"/");
#endif
         strcat(ptr_w,ptr_path);
         strcpy(ptr_path,ptr_w);
         if(DEBUG) printf("DEBUG: Final report JPEG name: %s\n",ptr_path);
         if(DEBUG) printf("DEBUG: RENAME-A: %s to %s\n",ptr_j,ptr_path);
         rename(ptr_j,ptr_path);

         /* get any data after the </ansysgr> tag */
         j=ptr_c - ptr_w + 10;
         if(netscape) j=j + 3;
         ptr_c=&ptr_in[j]; /* use original line, not lower case */
         to_ansys_output(ptr_c);
         return;
      }

      /* </ansysgr> tag is at middle or end of line */
      else
      {
         j=ptr_c - ptr_w;
         strcpy(ptr_w,ptr_in); /* use original line, not lower case */
         ptr_w[j]='\n';
         ptr_w[j+1]='\0';
         send_to_scratch(ptr_w);

         /* run the scratch file */
         scratch_open=2;
         send_to_scratch(" ");

         strcpy(ptr_w,"<img src=\"");
         strcat(ptr_w,ptr_path);
         strcat(ptr_w,"\" alt=\"\" width=483 height=363>");
         if (DEBUG) printf("DEBUG: IMG: %s\n",ptr_w);
         to_ansys_output(ptr_w);

         /* move jpeg file to report directory */
#ifdef WIN32
         strcpy(ptr_w,".\\");
#else
         strcpy(ptr_w,"./");
#endif
         strcat(ptr_w,ptr_dir);
#ifdef WIN32
         strcat(ptr_w,"\\");
#else
         strcat(ptr_w,"/");
#endif
         strcat(ptr_w,ptr_path);
         strcpy(ptr_path,ptr_w);
         if(DEBUG) printf("DEBUG: Final report JPEG name: %s\n",ptr_path);
         if(DEBUG) printf("DEBUG: RENAME-B: %s to %s\n",ptr_j,ptr_path);
         rename(ptr_j,ptr_path);

         /* get any data after the </ansysgr> tag */
         j=j + 10;
         if(netscape) j=j + 3;
         ptr_c=&ptr_in[j];
         strcpy(ptr_w,ptr_c); /* use original line, not lower case */
         to_ansys_output(ptr_w);
         return;
      }
   }
   
   /* send to ansys for processing */
   send_to_scratch(ptr_in);   /* use original line, not lower case */

   return;
}

void to_ansys_output(char* line)
{
   /*
      PURPOSE: 
      send a command to ansys output

      INPUT:
      char *line   command string to write to ansys output
   */

   if(line[0] != '\0' && line[0] != '\n')
   {
      if(DEBUG1) printf("DEBUG: to ANSYS output: %s\n",line);
      cAns_printff(line);
   }
   return;
}

void send_to_ansys(char* line)
{
   /*
      PURPOSE: 
      send a command to ansys to execute.

      INPUT:
      char *line   command string to execute
   */
   int  int_line[256];
   int i,j;

   j=strlen(line);
   /*
   for(i=0; i<256; i++)
      int_line[i]=(int)line[i];
   htmlgui_(&j,int_line);
   */
   if(DEBUG1) printf("DEBUG: to ANSYS input: %s\n",line); 
   cAnsSendCommand(line); 
   return;
}

void send_to_scratch(char* line)
{
   /*
      PURPOSE: 
      send a command to a scratch file

      INPUT:
      char *line   command string to write to scratch file
   */
   char tmpline[81];
   /* try to open the scratch file */
   if(scratch_open == 0)
   {
      if( (fp_scratch=fopen("htmlscratch","wt")) == NULL)
      {
         printf("File: %s was NOT opened !\a\n", "htmlscratch");
         return;
      }
      else
      {
         if(DEBUG1) printf("DEBUG: open scratch file: %s\n","htmlscratch"); 
         fputs(line,fp_scratch);
         scratch_open=1;
      }
      return;
   }

   if(scratch_open == 1)
   {
      if(DEBUG1) printf("DEBUG: to scratch file: %s\n",line); 
      fputs(line,fp_scratch);
      return;
   }

   if(scratch_open == 2 )
   {
      /* put a /rmtmp command on scratch file */
      strcpy(tmpline,"/rmtmp");
      fputs(tmpline,fp_scratch);
      fputs(line,fp_scratch);
      fclose(fp_scratch);
      send_to_ansys(")/nopr");
      send_to_ansys(")/input,htmlscratch");
      scratch_open=0;
      return;
   }

   return;
}


/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */
/* =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~= */
/* ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ */

/*
   Written by: Greg Sharpe   6/9/98

   Process an Ansys command of the form:
   ~evl,'%3.0f',node,1,loc,x
   Where field 2 is a "C" format specifer
   and fields 3+ are an Ansys "*get" specifier.
*/
int evl_get(char* work)
{
  double  dblValue;
  int     i, intType, retval;
  char    strMsg[9];
  char wrkline[256];
  char tokline[256];
  char outStr[20];
  char *params, *fmat, *iptr;
  const char* delim = {","};
  void* zip = 0;

/* start at arguments */
  params = &(work[5]);
  strcpy(tokline,params);

/* strip off the format */
  fmat = strtok(tokline,delim);

/* find the stuff just past the format */
  for(iptr=params;*iptr!=',';iptr++);
  iptr++;
  strcpy(wrkline,iptr);
  retval = cAnsGetValue(wrkline,&dblValue,strMsg,&intType);

/* strip off the single quotes (') (39) */
  for(iptr=fmat;*iptr!='\0';iptr++)
  {
    if (*iptr==39) *iptr = 32;
    *iptr = tolower(*iptr);
  }

  if (intType == 1)
  {
    sprintf(outStr,fmat,dblValue);
    to_ansys_output(outStr);
  }
  else if (intType == 2)
  {
    if(strlen(strMsg) > 0)
    {
       sprintf(outStr,"%s",strMsg);
       to_ansys_output(outStr);
    }
  }
  share_error(1002,zip);
  return(0);
}

/*
   Written by: Greg Sharpe   6/11/98
   Purpose: get name of next Ansys JPG file
*/
int getnextjpg(char* work)
{
   char name[80];
   query_jpg(name);
   if(DEBUG) printf("DEBUG: jpg: %s\n", name);
   strcpy(work,name);
   return(0);
}

/*
   Written by: Greg Sharpe   6/11/98
   Purpose: get name of next Ansys JPG file
*/
void query_jpg(char* jname)
{
   int   i, nchf, ifile[80];
   int jkey;
   char  txt_fname[80];
   
   nchf = 1;
   jkey = 7;
   ifile[0] = 32;
   ifile[1] = 0;
   psfnam_(&jkey, &nchf, ifile);
   for (i=0; i<nchf; i++) txt_fname[i] = ifile[i];
   txt_fname[nchf] = 0;
   if(DEBUG) printf("DEBUG: query_jpg  jpg: %s\n", txt_fname);
   strcpy(jname,txt_fname);
   return;
}
