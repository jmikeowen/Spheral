#include "libs.hh"
#include "classes.hh"
#include "headers.hh"
namespace FractalSpace
{
  void dens_to_slices(Group& group,Fractal_Memory& mem,Fractal& frac)
  {
    int FractalRank=mem.p_mess->FractalRank;
    int FractalNodes=mem.p_mess->FractalNodes;
    ofstream& FF=mem.p_file->DUMPS;
    int zoom=Misc::pow(2,frac.get_level_max());
    int length_1=frac.get_grid_length();
    int length_S=length_1;
    int length_S2=0;
    bool period=frac.get_periodic();
    if(!period)
      {
	length_1++;
	length_S*=2;
	length_S2=length_S+2;
      }

    vector <int> pos_point(3);
    vector <int>counts_in;
    vector <int>counts_out;
    vector <vector <int> > dataI_out;
    vector <vector <double> > dataR_out;
    vector <int> dataI_in;
    vector <double> dataR_in;
    counts_out.assign(FractalNodes,0);
    for(vector <Point*>::const_iterator point_itr=group.list_points.begin();point_itr != group.list_points.end();++point_itr)
      {
	Point* p_point=*point_itr;
	if(p_point->get_passive_point())
	  continue;
	if(!p_point->get_mass_point())
	  continue;
	p_point->get_pos_point(pos_point);
	int nx=(pos_point[0]/zoom+length_1) % length_1;
	int S=mem.p_mess->WhichSlice[nx];
	counts_out[S]++;
      }
    vector <int>maxSR;
    mem.p_mess->MAX_Things_To_Send_Receive_I(counts_out,counts_in,maxSR);
    int maxOUT=maxSR[0]*2;
    int maxIN=maxSR[1]*2;
    int maxINOUT=max(maxOUT,maxIN);
    int maxIO=6000000;
    counts_in.clear();
    counts_out.clear();
    //
    int LOOPS=(maxINOUT-1)/maxIO+1;
    FF << " LOOPS " << maxIO << " " << maxOUT << " " << maxIN << " " << LOOPS << "\n";
    vector <double>times;
    for(int LOOP=0;LOOP<LOOPS;LOOP++)
      {
	times.push_back(mem.p_mess->Clock());
	dataI_out.clear();
	dataR_out.clear();
	dataI_in.clear();
	dataR_in.clear();
	dataI_out.resize(FractalNodes);
	dataR_out.resize(FractalNodes);
	counts_out.assign(FractalNodes,0);
	counts_in.assign(FractalNodes,0);
	int loop_count=0;
	for(vector <Point*>::const_iterator point_itr=group.list_points.begin();point_itr != group.list_points.end();++point_itr)
	  {
	    if(loop_count % LOOPS == LOOP)
	      {
		Point* p_point=*point_itr;
		if(p_point->get_passive_point())
		  continue;
		if(!p_point->get_mass_point())
		  continue;
		p_point->get_pos_point(pos_point);
		int nx=(pos_point[0]/zoom+length_1) % length_1;
		int ny=(pos_point[1]/zoom+length_1) % length_1;
		int nz=(pos_point[2]/zoom+length_1) % length_1;
		int S=mem.p_mess->WhichSlice[nx];
		dataI_out[S].push_back(frac.where(nx,ny,nz,mem.p_mess->BoxS[S],mem.p_mess->BoxSL[S]));
		dataR_out[S].push_back(p_point->get_density_point());
		counts_out[S]++;
	      }
	    loop_count++;
	  }
	int how_manyI=-1;
	int how_manyR=-1;
	int integers=1;
	int doubles=1;
	mem.p_file->note(true," dens to slices a ");
	mem.p_mess->Send_Data_Some_How(0,counts_out,counts_in,integers,doubles,
				       dataI_out,dataI_in,how_manyI,
				       dataR_out,dataR_in,how_manyR);
	mem.p_file->note(true," dens to slices c ");
	dataI_out.clear();
	dataR_out.clear();      
	if(LOOP == 0)
	  {
	    mem.p_mess->create_potRS();
	    mem.p_mess->zeroRS(-frac.get_density_0());
	  }
	int counterIR=0;
	for(int FR=0;FR<FractalNodes;FR++)
	  {
	    for(int c=0;c<counts_in[FR];c++)
	      {
		int NN=dataI_in[counterIR];
		if(!period)
		  {
		    int nz=NN % length_S2;
		    int ny=(NN/length_S2) % length_S;
		    int nx=NN/(length_S2*length_S);
		    assert(nx < length_1);
		    assert(ny < length_1);
		    assert(nz < length_1);
		    NN=nz+(ny+nx*length_1)*length_1;
		  }
		mem.p_mess->potRS[NN]=dataR_in[counterIR];
		//	    FF << " DS " << counterIR << " " << dataI_in[counterIR] << " " << NN << " " << dataR_in[counterIR] << "\n";
		counterIR++;
	      }
	  }
      }
    times.push_back(mem.p_mess->Clock());
    fprintf(mem.p_file->PFTime," dens to slices ");
    for(int ni=0;ni<LOOPS;ni++)
      fprintf(mem.p_file->PFTime," %3d %8.3E ",ni,times[ni+1]-times[ni]);
    fprintf(mem.p_file->PFTime,"\n");
  }
}
