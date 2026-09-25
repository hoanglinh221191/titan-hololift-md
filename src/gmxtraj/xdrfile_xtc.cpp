/* -*- mode: c; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-
 *
 * Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "xdrfile.h"
#include "xdrfile_xtc.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

enum { FALSE, TRUE };

namespace {

constexpr double kFrameTimeTolerancePs = 1.0e-4;
constexpr int64_t kFrameEstimateSampleLimit = 4;

int skip_xtc_coordinates(XDRFILE *xd, int expected_natoms,
                         int64_t file_size) {
  float box[DIM * DIM]{};
  if (xdrfile_read_float(box, DIM * DIM, xd) != DIM * DIM)
    return exdrFLOAT;

  int coord_count = 0;
  if (xdrfile_read_int(&coord_count, 1, xd) != 1)
    return exdrINT;
  if (coord_count < 0 || coord_count != expected_natoms)
    return exdr3DX;

  if (coord_count <= 9) {
    float raw_coords[9 * DIM]{};
    const int raw_values = coord_count * DIM;
    return xdrfile_read_float(raw_coords, raw_values, xd) == raw_values
               ? exdrOK
               : exdrFLOAT;
  }

  float precision = 0.0f;
  int minint[DIM]{};
  int maxint[DIM]{};
  int smallidx = 0;
  if (xdrfile_read_float(&precision, 1, xd) != 1)
    return exdrFLOAT;
  if (xdrfile_read_int(minint, DIM, xd) != DIM ||
      xdrfile_read_int(maxint, DIM, xd) != DIM ||
      xdrfile_read_int(&smallidx, 1, xd) != 1)
    return exdrINT;

  int64_t opaque_bytes = 0;
  if (xdrfile_get_xtc_magic(xd) == XTC_NEW_MAGIC) {
    if (xdrfile_read_int64(&opaque_bytes, 1, xd) != 1)
      return exdrINT;
  } else {
    int legacy_opaque_bytes = 0;
    if (xdrfile_read_int(&legacy_opaque_bytes, 1, xd) != 1)
      return exdrINT;
    opaque_bytes = legacy_opaque_bytes;
  }
  if (opaque_bytes < 0 ||
      opaque_bytes > std::numeric_limits<int64_t>::max() - 3)
    return exdr3DX;

  const int64_t raw_coord_bytes =
      static_cast<int64_t>(expected_natoms) * DIM * sizeof(float);
  const int64_t max_reasonable_opaque_bytes =
      std::max<int64_t>(128LL * 1024LL * 1024LL, raw_coord_bytes * 8LL);
  if (opaque_bytes > max_reasonable_opaque_bytes)
    return exdr3DX;

  const int64_t padded_bytes = (opaque_bytes + 3) & ~int64_t(3);
  const int64_t payload_start = xdr_tell(xd);
  if (payload_start < 0 || payload_start > file_size ||
      padded_bytes > file_size - payload_start)
    return exdr3DX;

  return xdr_seek(xd, padded_bytes, SEEK_CUR);
}

bool frame_is_selected(double time_ps, double begin_time_ps, bool begin_set,
                       double end_time_ps, bool end_set, double dt_ps,
                       double &next_target_time) {
  if ((begin_set && time_ps < begin_time_ps) ||
      (end_set && time_ps > end_time_ps))
    return false;

  if (dt_ps <= 0.0)
    return true;

  if (next_target_time < -1.0e17)
    next_target_time = begin_set ? begin_time_ps : time_ps;
  if (time_ps < next_target_time - kFrameTimeTolerancePs)
    return false;

  while (next_target_time <= time_ps + kFrameTimeTolerancePs)
    next_target_time += dt_ps;
  return true;
}

} // namespace

int xtc_header(XDRFILE *xd, int *natoms, int *step, float *time,
                      mybool bRead) {
  int result, magic, n = 1;

  /* Note: read is same as write. He he he */
  magic = xdrfile_get_xtc_magic(xd);
  if (magic != XTC_MAGIC && magic != XTC_NEW_MAGIC)
    magic = XTC_MAGIC;
  if ((result = xdrfile_write_int(&magic, n, xd)) != n) {
    if (bRead)
      return exdrENDOFFILE;
    else
      return exdrINT;
  }
  if (magic != XTC_MAGIC && magic != XTC_NEW_MAGIC)
    return exdrMAGIC;
  xdrfile_set_xtc_magic(xd, magic);
  if ((result = xdrfile_write_int(natoms, n, xd)) != n)
    return exdrINT;
  if ((result = xdrfile_write_int(step, n, xd)) != n)
    return exdrINT;
  if ((result = xdrfile_write_float(time, n, xd)) != n)
    return exdrFLOAT;

  return exdrOK;
}

static int xtc_coord(XDRFILE *xd, int *natoms, matrix box, rvec *x, float *prec,
                     mybool bRead) {
  int result;

  /* box */
  result = xdrfile_read_float(box[0], DIM * DIM, xd);
  if (DIM * DIM != result)
    return exdrFLOAT;
  else {
    if (bRead) {
      result = xdrfile_decompress_coord_float(x[0], natoms, prec, xd);
      if (result != *natoms)
        return exdr3DX;
    } else {
      result = xdrfile_compress_coord_float(x[0], *natoms, *prec, xd);
      if (result != *natoms)
        return exdr3DX;
    }
  }
  return exdrOK;
}

int read_xtc_natoms(char *fn, int *natoms) {
  XDRFILE *xd;
  int step, result;
  float time;

  xd = xdrfile_open(fn, "r");
  if (NULL == xd)
    return exdrFILENOTFOUND;
  result = xtc_header(xd, natoms, &step, &time, TRUE);
  xdrfile_close(xd);

  return result;
}

int estimate_xtc_frame_count(const char *fn, double begin_time_ps,
                             mybool begin_set, double end_time_ps,
                             mybool end_set, double dt_ps,
                             int64_t *selected_frames) {
  if (fn == NULL || selected_frames == NULL)
    return exdrHEADER;
  *selected_frames = 0;

  std::error_code ec;
  const std::uintmax_t file_size_unsigned = std::filesystem::file_size(fn, ec);
  if (ec || file_size_unsigned >
                static_cast<std::uintmax_t>(std::numeric_limits<int64_t>::max()))
    return exdrFILENOTFOUND;
  const int64_t file_size = static_cast<int64_t>(file_size_unsigned);

  XDRFILE *xd = xdrfile_open(fn, "r");
  if (xd == NULL)
    return exdrFILENOTFOUND;

  int result = exdrOK;
  int expected_natoms = -1;
  int64_t first_frame_start = -1;
  int64_t sampled_end = -1;
  std::vector<double> sampled_times;
  sampled_times.reserve(static_cast<std::size_t>(kFrameEstimateSampleLimit));
  double next_target_time = -1.0e18;
  while (static_cast<int64_t>(sampled_times.size()) <
         kFrameEstimateSampleLimit) {
    const int64_t frame_start = xdr_tell(xd);
    if (frame_start == file_size)
      break;
    if (frame_start < 0 || frame_start > file_size) {
      result = exdrHEADER;
      break;
    }

    int natoms = 0;
    int step = 0;
    float time_ps = 0.0f;
    result = xtc_header(xd, &natoms, &step, &time_ps, TRUE);
    if (result != exdrOK) {
      if (result == exdrENDOFFILE)
        result = exdrHEADER;
      break;
    }
    if (expected_natoms < 0)
      expected_natoms = natoms;
    if (natoms != expected_natoms) {
      result = exdr3DX;
      break;
    }
    if (first_frame_start < 0)
      first_frame_start = frame_start;
    sampled_times.push_back(static_cast<double>(time_ps));

    const bool selected = frame_is_selected(
        static_cast<double>(time_ps), begin_time_ps, begin_set != 0,
        end_time_ps, end_set != 0, dt_ps, next_target_time);
    if (selected) {
      if (*selected_frames == std::numeric_limits<int64_t>::max()) {
        result = exdrNOMEM;
        break;
      }
      ++(*selected_frames);
    }

    result = skip_xtc_coordinates(xd, natoms, file_size);
    if (result != exdrOK)
      break;
    sampled_end = xdr_tell(xd);

    if (end_set && static_cast<double>(time_ps) > end_time_ps)
      break;
  }

  const bool exact_sample =
      result == exdrOK &&
      (sampled_end == file_size ||
       (end_set && !sampled_times.empty() &&
        sampled_times.back() > end_time_ps));
  if (result == exdrOK && !exact_sample) {
    if (sampled_times.size() < 2 || first_frame_start < 0 ||
        sampled_end <= first_frame_start) {
      result = exdrHEADER;
    } else {
      const long double sampled_bytes =
          static_cast<long double>(sampled_end - first_frame_start);
      const long double mean_frame_bytes =
          sampled_bytes / static_cast<long double>(sampled_times.size());
      const long double estimated_raw_frames_ld = std::ceil(
          static_cast<long double>(file_size - first_frame_start) /
          mean_frame_bytes);
      if (!std::isfinite(static_cast<double>(estimated_raw_frames_ld)) ||
          estimated_raw_frames_ld < 1.0L ||
          estimated_raw_frames_ld >
              static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        result = exdrHEADER;
      } else {
        const int64_t estimated_raw_frames =
            static_cast<int64_t>(estimated_raw_frames_ld);
        const double sampled_time_span =
            sampled_times.back() - sampled_times.front();
        const double frame_dt_ps =
            sampled_time_span /
            static_cast<double>(sampled_times.size() - 1);
        if (!std::isfinite(frame_dt_ps) || frame_dt_ps <= 0.0) {
          result = exdrHEADER;
        } else {
          *selected_frames = 0;
          next_target_time = -1.0e18;
          for (int64_t frame = 0; frame < estimated_raw_frames; ++frame) {
            const double time_ps = sampled_times.front() +
                                   static_cast<double>(frame) * frame_dt_ps;
            if (end_set && time_ps > end_time_ps)
              break;
            if (frame_is_selected(time_ps, begin_time_ps, begin_set != 0,
                                  end_time_ps, end_set != 0, dt_ps,
                                  next_target_time)) {
              ++(*selected_frames);
            }
          }
        }
      }
    }
  }

  const int close_result = xdrfile_close(xd);
  return result == exdrOK ? close_result : result;
}

int read_xtc(XDRFILE *xd, int natoms, int *step, float *time, matrix box,
             rvec *x, float *prec)
/* Read subsequent frames */
{
  int result;

  if ((result = xtc_header(xd, &natoms, step, time, TRUE)) != exdrOK)
    return result;

  if ((result = xtc_coord(xd, &natoms, box, x, prec, 1)) != exdrOK)
    return result;

  return exdrOK;
}


int write_xtc(XDRFILE *xd, int natoms, int step, float time, matrix box,
              rvec *x, float prec)
/* Write a frame to xtc file */
{
  int result;

  if (natoms > XTC_1995_MAX_NATOMS)
    xdrfile_set_xtc_magic(xd, XTC_NEW_MAGIC);
  else if (xdrfile_get_xtc_magic(xd) != XTC_NEW_MAGIC)
    xdrfile_set_xtc_magic(xd, XTC_MAGIC);

  if ((result = xtc_header(xd, &natoms, &step, &time, FALSE)) != exdrOK)
    return result;

  if ((result = xtc_coord(xd, &natoms, box, x, &prec, 0)) != exdrOK)
    return result;

  return exdrOK;
}
