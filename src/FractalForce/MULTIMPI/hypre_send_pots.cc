#include "libs.hh"
#include "classes.hh"
#include "headers.hh"
namespace FractalSpace
{
  void hypre_send_pots(Fractal_Memory& mem,vector <Point*>& hypre_points,vector <double>& potH)
  {
    FILE* PFH=mem.p_file->PFHypre;
    int HypreRank=mem.p_mess->HypreRank;
    int HypreNodes=mem.p_mess->HypreNodes;
    vector <int> counts_out(HypreNodes,0);
    vector <vector <int> > dataI_out(HypreNodes);
    vector <vector <double> > dataR_out(HypreNodes);
    vector <int> counts_in(HypreNodes);
    vector <int> dataI_in;
    vector <double> dataR_in;
    int how_manyI=-1;
    int how_manyR=-1;
    int integers=0;
    int doubles=1;
    fprintf(PFH," starting looking %d %d \n",mem.ij_countsB[HypreRank],potH.size());
    int mi=mem.ij_offsetsB[HypreRank];
    int HR=0;
    for(int ni=0;ni<mem.ij_countsB[HypreRank];ni++)
      {
	while(mem.ij_offsets[HR+1] <= mi)
	  HR++;
	counts_out[HR]++;
	dataR_out[HR].push_back(potH[ni]);	
	mi++;
      }
    /*
    mem.p_mess->How_Many_Things_To_Send_I(mem.p_mess->HypreWorld,counts_out,counts_in);
    mem.p_mess->Send_Data_Somewhere_No_Block(mem.p_mess->HypreWorld,counts_out,counts_in,integers,doubles,
				    dataI_out,dataI_in,how_manyI,
				    dataR_out,dataR_in,how_manyR);
    */
    mem.p_mess->Send_Data_Some_How(mem.p_mess->HypreWorld,counts_out,counts_in,integers,doubles,
				   dataI_out,dataI_in,how_manyI,
				   dataR_out,dataR_in,how_manyR);
    assert(how_manyR == mem.ij_counts[HypreRank]);
    dataI_out.clear();
    dataR_out.clear();
    fprintf(PFH," data sizeA %d %d %d \n",how_manyR,dataI_in.size(),dataR_in.size());
    for(int ni=0;ni<how_manyR;ni++)
      hypre_points[ni]->set_potential_point(dataR_in[ni]);
  }
}
