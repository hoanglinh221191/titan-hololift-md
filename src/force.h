#ifndef FORCE_H
#define FORCE_H
#include<utility>
#include<cmath>
#include"uti_func.h"
using namespace std;

struct force
{
	double Fx{}, Fy{}, Fz{};

	force() :Fx{}, Fy{}, Fz{} {}

	template<typename T>
	force(T&& fx, T&& fy, T&& fz) : Fx{ fx }, Fy{ fy }, Fz{ fz } {}

	force(const force& rhs) :Fx{ rhs.Fx }, Fy{ rhs.Fy }, Fz{ rhs.Fz } {}
	force(force&& rhs) :Fx{ std::move(rhs.Fx) }, Fy{ std::move(rhs.Fy) }, Fz{ std::move(rhs.Fz) } {}

	template<typename T1, typename T2, typename T3>
	void set(T1&& fx, T2&& fy, T3&& fz) { Fx = std::forward<T1>(fx); Fy = std::forward<T2>(fy); Fz = std::forward<T3>(fz); }

	void set(double* point, size_t size) {
		if (size >= 3)
		{
			Fx = point[0]; Fy = point[1]; Fz = point[2];
		}
	}

	force& add(double&& x, double&& y, double&& z)
	{
		Fx += x;
		Fy += y;
		Fz += z;
		return *this;
	}

	force operator+(const force& rhs) const
	{
		return force(this->Fx + rhs.Fx, this->Fy + rhs.Fy, this->Fz + rhs.Fz);
	}

	force operator-(const force& rhs) const
	{
		return force(this->Fx - rhs.Fx, this->Fy - rhs.Fy, this->Fz - rhs.Fz);
	}

	force operator+(const force&& rhs) const
	{
		return force(this->Fx + rhs.Fx, this->Fy + rhs.Fy, this->Fz + rhs.Fz);
	}

	force operator-(const force&& rhs) const
	{
		return force(this->Fx - rhs.Fx, this->Fy - rhs.Fy, this->Fz - rhs.Fz);
	}

	force& operator=(force&& rhs)
	{
		Fx = std::move(rhs.Fx);
		Fy = std::move(rhs.Fy);
		Fz = std::move(rhs.Fz);
		return *this;
	}
	force& operator=(const force& rhs)
	{
		Fx = rhs.Fx;
		Fy = rhs.Fy;
		Fz = rhs.Fz;
		return *this;
	}
	force& operator +=(const force& rhs)
	{
		Fx += rhs.Fx;
		Fy += rhs.Fy;
		Fz += rhs.Fz;
		return *this;
	}
	force& operator +=(force&& rhs)
	{
		Fx += std::move(rhs.Fx);
		Fy += std::move(rhs.Fy);
		Fz += std::move(rhs.Fz);
		return *this;
	}
	force& operator /=(const double di)
	{
		Fx /= di;
		Fy /= di;
		Fz /= di;
		return *this;
	}
	force& operator *=(const double op)
	{
		Fx *= op;
		Fy *= op;
		Fz *= op;
		return *this;
	}

	force operator/(const double di)
	{
		return force(Fx / di, Fy / di, Fz / di);
	}

	inline double modul() const
	{
		return sqrt(square(Fx) + square(Fy) + square(Fz));
	}
};

#endif // !FORCE_H
