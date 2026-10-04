#include"Multi_MOL.h"
#include"arguments.h"
#include"info.h"
#include"folder.h"
#include<cmath>
#include<fstream>
#include<iomanip>
int set_chainnum(Multi_MOL& pro_in, Multi_MOL& lig, arguments const & Arguments)
{
	pro_in.set_chainnum();
	if (Arguments.flag_nocal == false)
		lig.set_chainnum();
	return 0;
}

