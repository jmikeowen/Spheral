// Timer.hh
#ifndef TIMER_H
#define TIMER_H

#include "DBC.hh"

#ifdef USE_MPI
#include "mpi.h"
#include "Distributed/Communicator.hh"
#endif

#include <list>
#include <string>

// if TIMER is not defined, then we just have a stub class.
#ifdef TIMER

extern "C" {
  // these are for gettimeofday() wall-clock timing
#include <unistd.h>
#include <sys/time.h>

  // obviously, if you don't have PAPI installed on your system
  // (which requires kernel modification) then the counters wont 
  // work, so don't compile with -DPAPI
#ifdef PAPI
#include <papi.h>
#endif

}

#ifndef TIMER_COUNTER
#define TIMER_COUNTER 0
#endif


#define DIAGNOSTIC false 

class Timer {
  
public:
  
  Timer();                             // unmanaged
  Timer(const std::string&);                // root
  Timer(const std::string&, Timer&);        // managed
  Timer(const std::string&, Timer&, bool);  // diagnostic
  ~Timer();
  
  void setup();
  void start();
  void stop();
  void clear();

  double getTimeStampWC();
  double wc_time() { return accumulated_WCtime; }

#ifdef PAPI
  long long int papi_counter1() { return accumulated_counter1; }
  long long int papi_counter2() { return accumulated_counter2; }
#endif

  std::string Name() { return timer_name;  }
  
  long int Count() { return count; }

  static std::list<Timer*> TimerList;

  static void TimerSummary(const std::string fname = "time.table",
                           const bool printAllTimers = false);
  
private:
  
  //bool timer_on;    // State of timer, either on(true) or off(false)
  bool diagnostic;

  double accumulated_WCtime, last_WCtime_stamp;

  // wall-clock timer data
#ifndef USE_MPI  
  struct timeval tv;   //  Values from call to gettimeofday
  struct timezone tz;
#endif

  int ID;  
  std::string timer_name;
  Timer& Parent;

  double avgWC,  minWC,  maxWC;

#ifdef PAPI
  long long int values[2];
  long long int accumulated_counter1;
  long long int accumulated_counter2;

#endif

  long long int count;
};


#else


// stub Timer class
#include <string>
#include <iostream>

#define DIAGNOSTIC false

class Timer {

public:

  Timer() {}
  Timer(const std::string&) {}
  //Timer(const std::string&, int) {}
  Timer(const std::string&, Timer&) {}
  Timer(const std::string&, Timer&, bool) {}
  ~Timer() {}

  static std::list<Timer*> TimerList;

  inline void setup(){}
  inline void start(){}
  inline void stop(){}  
  inline void clear(){}

  inline double getTimeStampWC() {return 0.0;}
  inline double wc_time() {return 0.0;}

#ifdef PAPI
  inline long long int papi_counter1() {return 0;}
  inline long long int papi_counter2() {return 0;}
#endif

  inline std::string Name() {return NULL;}
  
  inline long int Count() {return 0;}
  
  static void TimerSummary(const std::string fname = "time.table",
                           const bool printAllTimers = false) {
    CONTRACT_VAR(fname);
    CONTRACT_VAR(printAllTimers);
    int rank;
#ifdef USE_MPI
    MPI_Comm_rank(Spheral::Communicator::communicator(), &rank);
#else
    rank=0;
#endif
    if(rank==0) {
      std::cout << " Timers Disabled.  No timing output written."  << std::endl;
    }
  }

};

#endif // TIMER

#endif // TIMER_H
