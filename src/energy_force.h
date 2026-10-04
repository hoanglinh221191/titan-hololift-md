#ifndef ENERGY_FORCE_H
#define ENERGY_FORCE_H
#include"miscellaneous.h"
#include<cmath>
namespace energy_force_val {
	struct value_energy_force
	{
		double E_elec{}, E_vdw{}, Modul_F_elec{}, Modul_F_vdw{}, Modul_F_total{};
		size_t sz{};
		value_energy_force() :E_elec{}, E_vdw{}, Modul_F_elec{}, Modul_F_vdw{}, Modul_F_total{}, sz{} {}
		value_energy_force(double e_elec, double e_vdw, force& f_elec, force& f_vdw) :
			E_elec{ e_elec }, E_vdw{ e_vdw }, Modul_F_elec{ f_elec.modul() }, Modul_F_vdw{ f_vdw.modul() }, Modul_F_total{ force(f_elec + f_vdw).modul() }
			, sz{ 1 } {}
		value_energy_force(const value_energy_force& rhs) :E_elec{ rhs.E_elec }, E_vdw{ rhs.E_vdw }, Modul_F_elec{ rhs.Modul_F_elec },
			Modul_F_vdw{ rhs.Modul_F_vdw }, Modul_F_total{ rhs.Modul_F_total }, sz{ 1 } {}
		value_energy_force(  energy_res_base& rhs) :E_elec{ rhs.e_e }, E_vdw{ rhs.e_v }, Modul_F_elec{ rhs.F_elec.modul() },
			Modul_F_vdw{ rhs.F_vdw.modul() }, Modul_F_total{ force(rhs.F_elec + rhs.F_vdw).modul() }, sz{ 1 } {}

		value_energy_force& operator+=(const energy_res_base& rhs);

		value_energy_force& operator+=(const value_energy_force& rhs);

		value_energy_force& operator/=(size_t size);

		value_energy_force operator-(const energy_res_base& rhs);

		value_energy_force operator-(const value_energy_force& rhs);

		template<typename T>
		value_energy_force multiply(T&& rhs)
		{
			value_energy_force result;
			result.E_elec = this->E_elec*rhs.E_elec;
			result.E_vdw = this->E_vdw*rhs.E_vdw;
			result.Modul_F_elec = this->Modul_F_elec*rhs.Modul_F_elec;
			result.Modul_F_vdw = this->Modul_F_vdw*rhs.Modul_F_vdw;
			result.Modul_F_total = this->Modul_F_total*rhs.Modul_F_total;
			return result;
		}

		value_energy_force multiply(double&& rhs)
		{
			value_energy_force result;
			result.E_elec *= rhs;
			result.E_vdw *= rhs;
			result.Modul_F_elec *= rhs;
			result.Modul_F_vdw *= rhs;
			result.Modul_F_total *= rhs;
			return result;
		}
		value_energy_force& sqrt()
		{
			E_elec = std::sqrt(E_elec);
			E_vdw = std::sqrt(E_vdw);
			Modul_F_elec = std::sqrt(Modul_F_elec);
			Modul_F_vdw = std::sqrt(Modul_F_vdw);
			Modul_F_total = std::sqrt(Modul_F_total);
			return *this;
		}
	};
}
struct multi_value_energy_force
{
	vector<energy_force_val::value_energy_force> values;
};
#endif // !ENERGY_FORCE_H

