// Timer.cc

#ifdef TIMER 

#include <list>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
using std::list;
using std::string;
using std::pair;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::abs;

#include "Timer.hh"
#include "DBC.hh"

// Must initialize the static list defined in Timer.hh
// list<Timer*> Timer::TimerList(0); 


static int ID_counter = 0;

#ifdef PAPI
static float CPU_MHZ = 0.1;

// These are just helper functions that call PAPI functions
//   Only used here and only for making code easier to read.
//static void PAPI_initSingleEvent(const int event, int *tag);
static void PAPI_initDualEvent(const int event1, const int event2, 
			       int *tag);
//static void PAPI_initMultiEvent(const int event1, const int event2, 
//			const int event3, int *tag);
//static void PAPI_checkEvent(const int event);

static void PAPI_init(void);
static void PAPI_hwinfo(float &mhz);
//static void PAPI_initMultiplex(void);
static void PAPI_cleanup(int *i);
#endif

// These are helper functions to write the formatted output
//  summary.  Nothing special.
static void writeDoubleLineSeparator(FILE *out);
static void writeSingleLineSeparator(FILE *out);
static void writeTableHeader(FILE *out);
static void writeLineOfData(FILE *out,
			    const char *name,
			    const long int count,
			    const double percent,
			    const double wc_avg,
			    const double wc_min,
			    const double wc_max,
			    const double counter1,
			    const double counter2);

static void writeTableTotals(FILE *out, 
			     const double parent_avg,
			     const double table_avg_sum, 
			     const double table_min_sum,
			     const double table_max_sum);

// Timer construction for the root of the tree.  When called without a
// parent, it is assumed that the Timer is the root.
// We'll also assume this is the Timer class init call -- and will
// be called first -- or at least before any start/stops.
Timer::Timer(const string& name):
  timer_name(name),  Parent(*this)  {
   
  //cout << " Root Timer construction for = " << name << endl;

  // Make a list of pointers of all Managed Timers.
  TimerList.push_back(this);
  
  setup();
}

// Non-root Managed Timer construction
Timer::Timer(const string& name, Timer& parent):
  timer_name(name),  Parent(parent) {

  //cout << "   Timer construction for = " << name << endl;

  TimerList.push_back(this);

  setup();
}

// Diagnostic Non-root Managed Timer construction for separate table
Timer::Timer(const string& name, Timer& parent, bool):
  timer_name(name),  Parent(parent) {

  //cout << "   Diagnostic Timer construction for = " << name << endl;

  TimerList.push_back(this);

  setup();
  diagnostic = true;
}


// Unmanaged Timer construction
Timer::Timer(): Parent(*this) {
  setup();
}


void Timer::setup() {

  count = 0;
  diagnostic = false;
  //timer_on = false;
  
  accumulated_WCtime = 0.0;
  ID = ID_counter;
  ID_counter++;

#ifdef PAPI
  //printf(" ID=%d\n", ID);

  // this only needs to be called once. 
  // and before any PAPI calls.  not sure where to put it.
  if(CPU_MHZ < 1.0) {
    PAPI_init();
    PAPI_hwinfo(CPU_MHZ);
  }
  
  //PAPI_initMultiplex();


  accumulated_counter1 = 0;
  accumulated_counter2 = 0;

  values[0] = 0;
  values[1] = 0;

  //PAPI_initSingleEvent(PAPI_TOT_CYC, &myEventSet);

  if(TIMER_COUNTER == 0) {
    PAPI_initDualEvent(PAPI_TOT_CYC, PAPI_FP_INS, &ID);
  } else if(TIMER_COUNTER == 1) {
    PAPI_initDualEvent(PAPI_L1_TCM,  PAPI_L2_TCM, &ID);
  } else {
    PAPI_initDualEvent(PAPI_TOT_INS, PAPI_BR_INS, &ID);
  }
  
  //PAPI_initMultiplex();
  //printf(" initting Timer %d\n", ID);
  //PAPI_initMultiEvent(PAPI_FP_INS, PAPI_TOT_CYC, PAPI_L1_TCM, &ID);

#endif

}

// Destructor for Managed and Unmanaged Timers.
Timer::~Timer() {
  //  cout << " Timer::~Timer() " << Name() << endl;
}


void Timer::start(void) {

  ++count; 
  last_WCtime_stamp  = getTimeStampWC();

#ifdef PAPI
#ifdef NDEBUG
  PAPI_start(ID);
#else
  assert(PAPI_start(ID) == PAPI_OK); 
#endif
  //printf(" start ID%d\n", ID);
#endif  
}

void Timer::stop(void) {

  accumulated_WCtime += getTimeStampWC() - last_WCtime_stamp;
#ifdef PAPI
#ifdef NDEBUG
  PAPI_stop(ID, values);
#else
  assert(PAPI_stop(ID, values) == PAPI_OK);
#endif
  accumulated_counter1 += values[0];
  accumulated_counter2 += values[1];
#endif  

}

void Timer::clear(void) {

  accumulated_WCtime = 0.;
#ifdef PAPI
  accumulated_counter1 = 0;
  accumulated_counter2 = 0;
#endif 

  count = 0;
}

double Timer::getTimeStampWC(){

#ifdef USE_MPI
  return( MPI_Wtime() );
#else    
  gettimeofday(&tv, &tz);
  //printf( "%d seconds, %d microseconds\n", tv.tv_sec, tv.tv_usec); 
  //cout << tvbuf->tv_sec << endl;
  //return(  ((double)((tvbuf->tv_sec)*1000000 + tvbuf->tv_usec))*1.0e-6 );

  // this doesn't always work cuz of potential integer overflow.
  //return( (tv.tv_sec * 1000000 + tv.tv_usec) * 1.0e-6 );

  // this is what MPICH uses for MPI_Wtime on machines
  // which have gettimeofday.
  return((double)tv.tv_sec + 0.000001 * (double)tv.tv_usec);
#endif
}

// This is the Summary for the Managed Timers.  If running in
// parallel, i first reduce all of the accumulated times for each
// timer and get the minimum, maximum, and average.  Then i go thru
// the list of timers and make a list of parent timers.  From there i
// can make for loops that step thru the tree and print out the
// results.
void Timer::TimerSummary( const std::string fname ) {

  int rank, number_procs;
#ifdef USE_MPI
  MPI_Comm_rank(Spheral::Communicator::communicator(), &rank);
  MPI_Comm_size(Spheral::Communicator::communicator(), &number_procs);
#else
  rank=0;
  number_procs=1;
#endif

  // printf("Rank %d, Procs %d, num timers %d\n", rank, number_procs, TimerList.size());

  list<Timer*>::iterator tli;  
  
  double wc;

  // go thru all Timers in static list and obtain the avg/min/max of
  // all processors onto rank0 -- obviously trivial for serial runs.
  for(tli=TimerList.begin() ; tli != TimerList.end() ; ++tli) {
      
    // printf(" Timer: %-16s Parent: %-16s diagnostic = %d\n",
    //    (*tli)->Name().c_str(), (*tli)->Parent.Name().c_str(), 
    //    (*tli)->diagnostic);
      
    wc = (*tli)->wc_time();
      
    if (number_procs == 1) {
      (*tli)->avgWC = wc;
      (*tli)->minWC = wc;
      (*tli)->maxWC = wc;

    } else {

#ifdef USE_MPI
         
      MPI_Reduce(&wc, &(*tli)->minWC, 1, MPI_DOUBLE, 
		 MPI_MIN, 0, Spheral::Communicator::communicator());
      MPI_Reduce(&wc, &(*tli)->maxWC, 1, MPI_DOUBLE, 
		 MPI_MAX, 0, Spheral::Communicator::communicator());

      double temp;
      MPI_Reduce(&wc, &temp, 1, MPI_DOUBLE, 
		 MPI_SUM, 0, Spheral::Communicator::communicator());

      if(rank==0) (*tli)->avgWC = temp/number_procs;

#endif       

    }
      
  }

  // go away if you aren't rank 0
  if (rank != 0) { return; }
   
   
  // Create a list of Parent Timers from the list of Timers
  list<Timer*> ParentList;
  list<Timer*>::iterator pti; 

  //cout << " size of TimerList = " << TimerList.size() << endl;

  // for each timer in entire Timer List
  for(tli=TimerList.begin() ; tli != TimerList.end() ; tli++) {

    //printf(" Timer #%3d: %-16s Parent: %-16s diagnostic = %d\n",
    //   (*tli)->ID,
    //   (*tli)->Name().c_str(), 
    //   (*tli)->Parent.Name().c_str(), 
    //   (*tli)->diagnostic);

    // add the Parent of this Timer to the Parent List
    bool add = true;
      
    // but first, make sure it isn't already in the Parent List
    // for each parent currently in Parent List
    for(pti=ParentList.begin() ; pti != ParentList.end() ; pti++) {
      if(*pti == &((*tli)->Parent) ) {
	add = false;      
	break;
      }
    }
      
    if(add) {
      ParentList.push_back( &((*tli)->Parent) );
      //cout << "  Adding Parent:" << (*tli)->Parent.Name() << endl;
    }
    //cout << " ParentList size = " << ParentList.size() << endl;
  }


   
  cout << " rank" << rank << " writing " << fname << endl;

  FILE *OUT;
  
  if(TIMER_COUNTER == 0) {
    OUT = fopen(fname.c_str(), "w");
  } else if (TIMER_COUNTER == 1) {
    OUT = fopen((fname + "1").c_str(), "w");
  } else {
    OUT = fopen((fname + "2").c_str(), "w");
  }
  if(OUT == NULL) {
    printf("problem opening output file in Timer\n");  
    exit(0); 
  }
  
  fprintf(OUT, "Number of Nodes: %d\n", number_procs );
  fprintf(OUT, "\n");
  
  
  double time_unit_factor;
  //time_unit_factor = 1.0/60.0; // for hours;
  time_unit_factor = 1.0; // for minutes;
  //time_unit_factor = 60.0; // for seconds;
  
  for(pti=ParentList.begin() ; pti != ParentList.end() ; pti++) {
    
    fprintf(OUT, "\n");

    writeTableHeader(OUT);
      
    double table_avg_sum     = 0.0;
    double table_min_sum     = 0.0;
    double table_max_sum     = 0.0;
    double table_percent;

    double wc_avg = (*pti)->avgWC*time_unit_factor;
    double wc_min = (*pti)->minWC*time_unit_factor;
    double wc_max = (*pti)->maxWC*time_unit_factor;

    double parent_avg = wc_avg;

    // Parent for each table
#ifdef PAPI
    writeLineOfData(OUT, (*pti)->Name().c_str(), (*pti)->Count(), -1, 
		    wc_avg, wc_min, wc_max,
		    (double)(*pti)->papi_counter1(), 
		    (double)(*pti)->papi_counter2());
#else
    writeLineOfData(OUT, (*pti)->Name().c_str(), (*pti)->Count(), -1,
		    wc_avg, wc_min, wc_max, 0, 0);
#endif

    writeSingleLineSeparator(OUT);

    for(tli=TimerList.begin() ; tli != TimerList.end() ; tli++) {
      
      if ((*tli)->diagnostic) continue;
      
      if( &((*tli)->Parent) == *pti && *tli != &((*tli)->Parent) ) {

  	wc_avg = (*tli)->avgWC*time_unit_factor;
  	wc_min = (*tli)->minWC*time_unit_factor;
  	wc_max = (*tli)->maxWC*time_unit_factor;
	
 	if(parent_avg  > 0.) {
 	  table_percent  = wc_avg/parent_avg*100.0;
 	} else {
 	  table_percent = 0.;
	}

  	table_avg_sum  += wc_avg;	
  	table_min_sum  += wc_min;	
  	table_max_sum  += wc_max;	

	// Children
#ifdef PAPI
	writeLineOfData(OUT, (*tli)->Name().c_str(), (*tli)->Count(), 
			table_percent, wc_avg, wc_min, wc_max,
			(double)(*tli)->papi_counter1(), 
			(double)(*tli)->papi_counter2());	
#else
	writeLineOfData(OUT, (*tli)->Name().c_str(), (*tli)->Count(), 
			table_percent, wc_avg, wc_min, wc_max, 0, 0);
#endif
      }
      
    } // end of table loop for this parent

    writeTableTotals(OUT, parent_avg, table_avg_sum, 
		     table_min_sum,  table_max_sum);
    
    
    fprintf(OUT,"\n\n");
  } // end of parent loop



  int Ndiag=0;
  double largestTime = -1.0;
  Timer *DiagnosticParent=NULL;
    
  for(tli=TimerList.begin() ; tli != TimerList.end() ; tli++) {    
    if ((*tli)->diagnostic == true) Ndiag++;
    if ((*tli)->avgWC > largestTime) {
      DiagnosticParent = *tli;
      largestTime = (*tli)->avgWC;
    }
    //cout << DiagnosticParent->Name() << endl;
  }
  
  if(Ndiag != 0) {

    fprintf(OUT, "\n\n Diagnostic Table \n");
    
    writeTableHeader(OUT);
    
    double table_avg_sum     = 0.0;
    double table_min_sum     = 0.0;
    double table_max_sum     = 0.0;
    double table_percent;
    
    double wc_avg = DiagnosticParent->avgWC*time_unit_factor;
    double wc_min = DiagnosticParent->minWC*time_unit_factor;
    double wc_max = DiagnosticParent->maxWC*time_unit_factor;
    
    double parent_avg = wc_avg;
    
    // Parent for each table
#ifdef PAPI
    writeLineOfData(OUT, DiagnosticParent->Name().c_str(), 
		    DiagnosticParent->Count(), -1,
		    wc_avg, wc_min, wc_max,
		    (double)DiagnosticParent->papi_counter1(), 
		    (double)DiagnosticParent->papi_counter2());
#else
    writeLineOfData(OUT, DiagnosticParent->Name().c_str(), 
		    DiagnosticParent->Count(), -1, 
		    wc_avg, wc_min, wc_max, 0, 0);
#endif
    
    writeSingleLineSeparator(OUT);  
    
    for(tli=TimerList.begin() ; tli != TimerList.end() ; tli++) {

      if ((*tli)->diagnostic == false) continue;

      //if( &((*tli)->Parent) == DiagnosticParent 
      //  && *tli != &((*tli)->Parent) ) {

	wc_avg = (*tli)->avgWC*time_unit_factor;
	wc_min = (*tli)->minWC*time_unit_factor;
	wc_max = (*tli)->maxWC*time_unit_factor;
	
	if(parent_avg  > 0.) {
	  table_percent  = wc_avg/parent_avg*100.0;
	} else {
	  table_percent = 0.;
	}
	
	table_avg_sum  += wc_avg;	
	table_min_sum  += wc_min;	
	table_max_sum  += wc_max;	
	
	// Children
#ifdef PAPI
	writeLineOfData(OUT, (*tli)->Name().c_str(), (*tli)->Count(), 
			table_percent, wc_avg, wc_min, wc_max,
			(double)(*tli)->papi_counter1(), 
			(double)(*tli)->papi_counter2());
	
#else
	writeLineOfData(OUT, (*tli)->Name().c_str(), (*tli)->Count(), 
			table_percent, wc_avg, wc_min, wc_max, 0, 0);
#endif
        //}
      
    } // end of table loop for this parent
    
    
    writeTableTotals(OUT, parent_avg, table_avg_sum, 
		     table_min_sum,  table_max_sum);
    
    fprintf(OUT,"\n\n");
  }



  fclose(OUT);

#ifdef PAPI

  // not sure how important this cleanup and shutdown is...
  for(int i=0; i<ID_counter; i++) {
    //printf(" cleaning up event %d\n", i);
    PAPI_cleanup(&i);
  }

  PAPI_shutdown();
#endif
}


static void writeSingleLineSeparator(FILE *out) {
#ifdef PAPI  
  fprintf(out, "--------------------------------------------------------------------");
  fprintf(out, "----------------------------------------------------------\n");
#else
  fprintf(out, "--------------------------------------------------------------------");
  fprintf(out, "------------------------------\n");
#endif
}

static void writeDoubleLineSeparator(FILE *out) {
#ifdef PAPI
  fprintf(out, "====================================================================");
  fprintf(out, "==========================================================\n");
#else
  fprintf(out, "====================================================================");
  fprintf(out, "==============================\n");
#endif
}


static void writeTableHeader(FILE *out) {


#ifdef PAPI
  fprintf(out, "                                                                     ");
  fprintf(out, "wall-clock time (sec)\n");
  fprintf(out, "                                                                   ");
  fprintf(out, "________________________\n");
  
  //            12345678901234567890
  fprintf(out, "      Totals                  ");
  fprintf(out, "            WC per.");
  fprintf(out, "     count");

  fprintf(out, "        avg ");
  fprintf(out, "        min ");
  fprintf(out, "       max  ");

  if(TIMER_COUNTER==0) {
    fprintf(out, "   cycles     FP INS     mflops");
  } else if(TIMER_COUNTER==1) {
    fprintf(out, "   L1 TCM     L2 TCM     L2/L1");
  } else {
    fprintf(out, "   TOT INS    BR INS     BR/TOT");
  }
#else
  fprintf(out, "                                                                     ");
  fprintf(out, "wall-clock time (sec)\n");
  fprintf(out, "                                                                 ");
  fprintf(out, "______________________________\n");
  
  //            12345678901234567890
  fprintf(out, "      Totals    ");
  fprintf(out, "            WC per.              ");
  fprintf(out, "     count");

  fprintf(out, "        avg  ");
  fprintf(out, "       min  ");
  fprintf(out, "       max  ");
#endif

  fprintf(out, "\n");

  writeDoubleLineSeparator(out);  

}

static void writeTableTotals(FILE *out, 
			     const double parent_avg,
			     const double table_avg_sum, 
			     const double table_min_sum,
			     const double table_max_sum) {

  writeDoubleLineSeparator(out);


  double table_percent;
  if(parent_avg  > 0.) {
    table_percent = table_avg_sum/parent_avg * 100.0;
  } else {
    table_percent = 0.;
  }

#ifdef PAPI
  fprintf(out, "  table totals:                          (%6.2f%%) %8s %10.2f [%10.2f,%10.2f]\n",
	  table_percent, " ", 
	  table_avg_sum, table_min_sum, table_max_sum); 
#else
  fprintf(out, "  table totals:                          (%6.2f%%) %8s %10.2f [%10.2f, %10.2f]\n", 
	  table_percent, " ", 
	  table_avg_sum, table_min_sum, table_max_sum); 
#endif

}

static void writeLineOfData(FILE *out,
			    const char *name,
			    const long int count,
			    const double percent,
			    const double wc_avg,
			    const double wc_min,
			    const double wc_max,
			    const double counter1,
			    const double counter2) {

  CONTRACT_VAR(counter1);
  CONTRACT_VAR(counter2);

#ifdef PAPI

  const float VALID_PROC_TIME = 1.0e-4;

  float proc_time = 1.0;
  float derived_counter = 0.0;

  if(TIMER_COUNTER == 0) {
    if(CPU_MHZ <= 0.0) CPU_MHZ = 1.0;
    proc_time = counter1/(CPU_MHZ*1.0e6);
    if(proc_time > VALID_PROC_TIME) {
      derived_counter = counter2/(proc_time*1.0e6);
    } else {
      derived_counter = 0.0;
    }
  } else if (TIMER_COUNTER == 1) {    
    if(counter1 > 0) {
      derived_counter = counter2/counter1;
    } else {
      derived_counter = 0.0;
    }
  } else {
    if(counter1 > 0) {
      derived_counter = counter2/counter1;
    } else {
      derived_counter = 0.0;
    }
  }
  
  
  if(percent < 0) {
    fprintf(out, "%-40.40s           %8ld %10.2f [%10.2f,%10.2f]  %10.3e %10.3e %8.2f\n",
	    name, count, wc_avg, wc_min, wc_max,
	    counter1, counter2, 
	    derived_counter);
  } else {
    fprintf(out, "%-40.40s (%6.2f%%) %8ld %10.2f [%10.2f,%10.2f]  %10.3e %10.3e %8.2f\n",
	    name, percent, count, wc_avg, wc_min, wc_max,
	    counter1, counter2, 
	    derived_counter);
  }
#else
  if(percent < 0) {
    fprintf(out, "%-40.40s           %8ld %10.2f [%10.2f, %10.2f]\n",
	    name, count, wc_avg, wc_min, wc_max);  
  } else {
    fprintf(out, "%-40.40s (%6.2f%%) %8ld %10.2f [%10.2f, %10.2f]\n",
	    name, percent, count, wc_avg, wc_min, wc_max);  
  }
#endif
  
}

#ifdef PAPI
// void PAPI_initSingleEvent(const int event, int *tag) {
//   int retval;
  
//   retval = PAPI_query_event(event);
//   assert(retval == PAPI_OK);

//   retval = PAPI_create_eventset(tag);
//   assert(retval == PAPI_OK);

//   retval = PAPI_add_event(tag, event);
//   assert(retval == PAPI_OK);
// }

static void PAPI_initDualEvent(const int event1, const int event2, 
			       int *tag) {
  int retval;

  retval = PAPI_query_event(event1);
  assert(retval == PAPI_OK);

  retval = PAPI_query_event(event2);
  assert(retval == PAPI_OK);

  retval = PAPI_create_eventset(tag);
  assert(retval == PAPI_OK);

  retval = PAPI_add_event(tag, event1);
  assert(retval == PAPI_OK);

  retval = PAPI_add_event(tag, event2);
  assert(retval == PAPI_OK);


}


// static void PAPI_initMultiEvent(const int event1, const int event2, 
// 				const int event3, int *tag) {

//   int retval;

//   retval = PAPI_query_event(event1);
//   assert(retval == PAPI_OK);
//   retval = PAPI_query_event(event2);
//   assert(retval == PAPI_OK);
//   retval = PAPI_query_event(event3);
//   assert(retval == PAPI_OK);

//   retval = PAPI_create_eventset(tag);
//   assert(retval == PAPI_OK);

//   retval = PAPI_add_event(tag, event1);
//   assert(retval == PAPI_OK);
//   retval = PAPI_add_event(tag, event2);
//   assert(retval == PAPI_OK);

//   retval = PAPI_set_multiplex(tag);
//   assert(retval == PAPI_OK);

//   retval = PAPI_add_event(tag, event3);
//   assert(retval == PAPI_OK);
// }


// static void PAPI_checkEvent(const int event) {
//   int tag = PAPI_NULL;
//   long int value = -1;
//   int retval;

//   if( (retval = PAPI_query_event(event)) != PAPI_OK ) {
//     printf(" problem with PAPI_query_event retval=%d\n", retval);    
//     return;
//   }
//   if( (PAPI_create_eventset(&tag) != PAPI_OK) ) {
//     printf(" problem with PAPI_create_eventset\n");
//     return;
//   }
  
//   if( (PAPI_add_event(&tag, event) != PAPI_OK) ) {
//     printf(" problem with PAPI_add_event\n");
//     return;
//   }
//   if( (PAPI_start(tag) != PAPI_OK) ) {
//     printf(" problem with PAPI_start\n");
//     return;
//   }
//   if( (PAPI_stop(tag, &value) != PAPI_OK) ) {
//     printf(" problem with PAPI_stop\n");
//     return;
//   }
//   printf(" event looks ok.  value=%ld\n", value);
// }



static void PAPI_init(void) {
  int retval;
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  assert(retval == PAPI_VER_CURRENT);
}

static void PAPI_hwinfo(float &mhz) {
  const PAPI_hw_info_t *hwinfo = NULL;
  hwinfo = PAPI_get_hardware_info();
  assert( hwinfo != NULL );

  printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);

  mhz = hwinfo->mhz;
}

// static void PAPI_initMultiplex(void) {
// #ifdef NDEBUG  // no debug
//   PAPI_multiplex_init();
// #else
//   assert(PAPI_multiplex_init() == PAPI_OK);
// #endif
// }

static void PAPI_cleanup(int *i) {
#ifdef NDEBUG  // no debug
  PAPI_cleanup_eventset(i);   
#else
  assert(PAPI_cleanup_eventset(i) == PAPI_OK);   
#endif
}

#endif // on PAPI

#else
//#include "Timer.H"
// Must initialize the static list defined in Timer.hh
//list<Timer*> Timer::TimerList(0); 

#endif  // TIMER
