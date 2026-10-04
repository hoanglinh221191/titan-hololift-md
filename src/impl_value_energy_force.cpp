#include"energy_force.h"
energy_force_val::value_energy_force& energy_force_val::value_energy_force::operator+=(const energy_res_base& rhs)
{
	E_elec += rhs.e_e;
	E_vdw += rhs.e_v;
	Modul_F_elec += rhs.F_elec.modul();
	Modul_F_vdw += rhs.F_vdw.modul();
	Modul_F_total += force(rhs.F_elec + rhs.F_vdw).modul();
	++sz;
	return *this;
}

energy_force_val::value_energy_force& energy_force_val::value_energy_force::operator+=(const value_energy_force& rhs)
{
	E_elec += rhs.E_elec;
	E_vdw += rhs.E_vdw;
	Modul_F_elec += rhs.Modul_F_elec;
	Modul_F_vdw += rhs.Modul_F_vdw;
	Modul_F_total += rhs.Modul_F_total;
	++sz;
	return *this;
}

energy_force_val::value_energy_force& energy_force_val::value_energy_force::operator/=(size_t size)
{
	E_elec /= size;
	E_vdw /= size;
	Modul_F_elec /= size;
	Modul_F_vdw /= size;
	Modul_F_total /= size;
	return *this;
}

energy_force_val::value_energy_force energy_force_val::value_energy_force::operator-(const energy_res_base& rhs)
{
	value_energy_force result;
	result.E_elec = this->E_elec - rhs.e_e;
	result.E_vdw = this->E_vdw - rhs.e_v;
	result.Modul_F_elec = this->Modul_F_elec - rhs.F_elec.modul();
	result.Modul_F_vdw = this->Modul_F_vdw - rhs.F_vdw.modul();
	result.Modul_F_total = this->Modul_F_total - force(rhs.F_elec + rhs.F_vdw).modul();
	return result;
}

energy_force_val::value_energy_force energy_force_val::value_energy_force::operator-(const value_energy_force& rhs)
{
	value_energy_force result;
	result.E_elec = this->E_elec - rhs.E_elec;
	result.E_vdw = this->E_vdw - rhs.E_vdw;
	result.Modul_F_elec = this->Modul_F_elec - rhs.Modul_F_elec;
	result.Modul_F_vdw = this->Modul_F_vdw - rhs.Modul_F_vdw;
	result.Modul_F_total = this->Modul_F_total - rhs.Modul_F_total;
	return result;
}

