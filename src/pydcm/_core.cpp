// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fuli Wu
//
// pydcm._core — nanobind extension: decode DICOM pixels to a NumPy array.
//
// A THIN wrapper over dcmcore/dcmbase — it reinvents nothing. dcmcore (split out
// of libdcm) already decodes EVERY transfer syntax (JPEG/J2K/HTJ2K/JLS/JXL/RLE),
// single- and multi-frame, and applies the modality LUT so the float path is
// HU; dcmbase::render exposes both that float and the native integer pixels.
// This file only marshals dcmcore's already-decoded output into NumPy:
//
//   rescale=False → extract_raw_frame  → native stored integers (lossless)
//   rescale=True  → extract_frame      → dcmcore's HU/rescaled float (GRAY32F),
//                                        per-frame rescale handled internally
//
// Geometry/rescale metadata is read straight off the decoder's dicom_info_t
// (it parsed those tags already) — no DICOM tag parsing here either.

#include <nanobind/nanobind.h>
#include <dcmcore/dicom_uid.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <dcmbasexx/render.hpp>
#include <dcmbasexx/dict.hpp>      // native union dictionary (keyword↔tag↔VR/VM)
#include <dcmbasexx/edit.hpp>      // byte-verbatim Part-10 editor (production save_as: keeps TS + pixels)
#include <dcmbasexx/deident.hpp>   // deident::session — conformant PS3.15 Annex E de-identification (native engine)
#include <dcmbasexx/pixanon.hpp>   // pixanon::clean — burned-in pixel-data blackout (113101 Clean Pixel Data)
#include <dcmbasexx/uid.hpp>       // uid::mint — THE canonical UID generator (img2dcm/pdf2dcm/deident share it)
#include <dcmbasexx/content.hpp>   // content::to_json — structured-object semantic JSON (dcm2content)
#include <dcmbasexx/iod.hpp>       // iod::validate — IOD/module Type-1/2 conformance (dcmvalidate core)
#include <dcmbasexx/dicomdir.hpp>  // dicomdir::build — Media Storage Directory (DICOMDIR) writer
#include <dcmbasexx/rt.hpp>        // rt::read_dose — RT Dose grid engine (scaling/geometry/DVH in C++)
#include <dcmbasexx/encap.hpp>     // encap::encapsulate/extract — Encapsulated Document engine (dcmencap)
#include <dcmbasexx/radiomics.hpp> // radiomics::extract — IBSI features (dcmradiomics)
#include <dcmbasexx/dce.hpp>       // dce::fit_slice — DCE-MRI PK parameter maps
#include <dcmbase/dcm_dce.h>       // dce_fit / dce_forward / dce_signal_to_conc (single-curve C core)
#include <dcmbasexx/seg.hpp>       // seg::build — coded DICOM Segmentation writer (mkseg)
#include <dcmbasexx/sr.hpp>        // sr::build — generic Structured Report writer (mksr)
#include <dcmbasexx/ko.hpp>        // ko::build — Key Object Selection writer (mkkos)
#include <dcmbasexx/wsi.hpp>       // wsi::slide — whole-slide pyramid reader (pydcm.wsi)
#include <dcmbasexx/wsi_write.hpp> // wsi::build_pyramid — WSI authoring (pydcm.wsi.write_slide)
#include <dcmbasexx/legacy_converted.hpp> // legacy_converted::convert — Legacy Converted Enhanced CT/MR/PET
#include <dcmbasexx/ann.hpp>      // ann::read — Microscopy Bulk Simple Annotations reader
#include <dcmcore/dicom_sr.h>      // SR_VT_* / SR_REL_* / SR_GRAPHIC_*
#include <dcmbase/dcm_volume.h>    // vol_extract_coords / vol_discover / vol_assemble (3D/N-D engine)
#include <dcmbase/dcm_dti.h>       // dti_prepare / dti_fit / dti_eigen_batch / dti_compute_map
#include <dcmbase/dcm_nifti.h>     // dcm_nifti_write / dcm_nifti_read (DICOM volume <-> .nii[.gz])
#include <dcmbasexx/volume.hpp>    // volume::assemble — shared series→3D-volume helper (pydcm + dcm2nii CLI)
#include <dcmbasexx/transform.hpp> // transform::resample_to_spacing / normalize_zscore / argmax (pydcm.transforms)
#include <dcmbasexx/mosaic.hpp>    // mosaic::expand — Siemens mosaic → N de-tiled slices (CSA-driven)
#include <dcmbasexx/bids.hpp>      // bids::extract — BIDS JSON sidecar
#include <dcmcorexx/dicom_bridge.hpp>  // bridge::to_dicom_json / dataset_to_json (full element model)
#include <dcmcorexx/dcm_transcode.hpp> // transcode::part10 (compress to encapsulated lossless)
#include <dcmcorexx/dicom_part10.hpp>  // part10::peek_file_meta (ds.file_meta)
#include <dcmcorexx/dicom_dataset.hpp> // dataset::parse (raw ds.PixelData extraction)
#include <dcmcore/dicom_image.h>   // PF_* pixel formats
#include <dcmcore/dicom_info.h>    // dicom_info_t fields (rescale / spacing / orientation / UIDs)
#include <dcmcore/dicom_walk.h>    // dcm_sniff_naked_dataset (file_meta of naked datasets)

extern "C" {
#include <dcmbase/dcm_sr_export.h>      // sr_export_* — TID 1500 Measurement Report writer (the mkreport engine)
#include <dcmbase/dcm_paramap_export.h> // paramap_export_* — float Parametric Map writer (itkimage2paramap)
#include <dcmbase/dcm_rtdose_export.h>  // rtdose_export_* — RT Dose grid writer (the write side of pydcm.rt)
#include <dcmbase/dcm_seg_decode.h>     // seg_decode_labelmap / seg_decode_masks (segimage2itkimage)
#include <dcmbase/dcm_ps_export.h>      // ps_export_* — Grayscale Softcopy Presentation State writer
}
#include <dcmcore/dicom_segmentation.h> // seg_document_t / seg_segment_t (read_seg metadata)

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
namespace rdr = dcmbase::render;

namespace {

std::vector<std::byte> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamoff n = f.tellg();
    std::vector<std::byte> buf(static_cast<std::size_t>(n));
    f.seekg(0);
    if (n) f.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

std::string dtype_int(std::uint32_t bits, bool sgn) {
    return (sgn ? "int" : "uint") + std::to_string(bits);
}

// dlpack dtype for the rescaled (display) path, keyed on dcmcore's pixel_format.
struct fmt_info { std::uint8_t code; std::uint8_t bits; std::uint32_t samples; const char* name; };
fmt_info display_fmt(std::uint8_t pf, std::uint16_t spp) {
    using c = nb::dlpack::dtype_code;
    switch (pf) {
        case PF_GRAY32F: return {(std::uint8_t)c::Float, 32, 1, "float32"};
        case PF_RGB32F:  return {(std::uint8_t)c::Float, 32, 3, "float32"};
        case PF_RGB8:    return {(std::uint8_t)c::UInt,   8, 3, "uint8"};
        case PF_ARGB8:   return {(std::uint8_t)c::UInt,   8, 4, "uint8"};
        case PF_RGBA8:   return {(std::uint8_t)c::UInt,   8, 4, "uint8"};
        case PF_MASK8:   return {(std::uint8_t)c::UInt,   8, 1, "uint8"};
        case PF_MASK16:  return {(std::uint8_t)c::UInt,  16, 1, "uint16"};
        default:         return {(std::uint8_t)c::Float, 32, spp ? spp : 1u, "float32"};
    }
}

const char* photometric_name(std::uint8_t pi) {
    switch (pi) {
        case PI_MONOCHROME1: return "MONOCHROME1";
        case PI_MONOCHROME2: return "MONOCHROME2";
        case PI_PALETTE_COLOR: return "PALETTE COLOR";
        case PI_RGB: return "RGB";
        case PI_YBR_FULL: return "YBR_FULL";
        case PI_YBR_FULL_422: return "YBR_FULL_422";
        default: return "";
    }
}

std::string cstr(const char* s) { return s ? std::string(s) : std::string(); }

// Read the geometry/rescale fields dcmcore already parsed into the dict.
void add_meta(nb::dict& meta, const dicom_info_t* info) {
    meta["modality"]               = cstr(info->modality);
    meta["photometric"]            = photometric_name(info->photometric_interpretation);
    meta["series_instance_uid"]    = cstr(info->series_instance_uid);
    meta["sop_instance_uid"]       = cstr(info->sop_instance_uid);
    meta["study_instance_uid"]     = cstr(info->study_instance_uid);
    meta["frame_of_reference_uid"] = cstr(info->frame_of_reference_uid);
    meta["instance_number"]        = cstr(info->instance_number);
    meta["rescale_slope"]          = info->rescale_slope;
    meta["rescale_intercept"]      = info->rescale_intercept;
    meta["window_center"]          = info->window_center;
    meta["window_width"]           = info->window_width;
    meta["slice_thickness"]        = info->slice_thickness;
    nb::list ps;  ps.append(info->pixel_spacing[0]);  ps.append(info->pixel_spacing[1]);
    meta["pixel_spacing"] = ps;                                  // [row, col] mm
    nb::list ipp; for (int i = 0; i < 3; ++i) ipp.append(info->origin[i]);
    meta["image_position_patient"] = ipp;                       // (0020,0032)
    nb::list iop; for (int i = 0; i < 6; ++i) iop.append(info->orientation[i]);
    meta["image_orientation_patient"] = iop;                    // (0020,0037)
}

// Shared core: decode from an in-memory Part-10 byte span. Both the path entry point
// (decode) and the in-memory one (decode_bytes) wrap this — identical native logic for
// every transfer syntax, so an in-memory dataset decodes the same as a file.
nb::object decode_span(std::span<const std::byte> bytes, int frame, bool rescale,
                       const std::string& src) {
    std::uint32_t rows = 0, cols = 0, spp = 0, ba = 0, bs = 0, planar = 0, frames = 0;
    std::uint8_t  dt_code = 0; bool sgn = false; std::string dtype_name;
    auto* out = new std::vector<std::byte>();
    std::optional<rdr::decoded> dec_owner;

    try {
        // Parse + per-frame decode are pure C++ on caller-owned bytes — run
        // them GIL-released so thread pools (DataLoader workers, batch
        // converters) decode in parallel instead of serialising on the GIL.
        nb::gil_scoped_release rel;
        dec_owner.emplace(bytes);
        rdr::decoded& dec = *dec_owner;
        if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + src);

        const std::uint32_t n = dec.frame_count();
        if (n == 0) throw std::runtime_error("no decodable image frames: " + src);
        if (frame < 0 || static_cast<std::uint32_t>(frame) > n)
            throw std::runtime_error("frame out of range (1.." + std::to_string(n) + "): " + src);

        const std::uint32_t first = frame > 0 ? static_cast<std::uint32_t>(frame) : 1;
        const std::uint32_t last  = frame > 0 ? static_cast<std::uint32_t>(frame) : n;
        frames = last - first + 1;

        for (std::uint32_t f = first; f <= last; ++f) {
            if (rescale) {
                auto fv = rdr::extract_frame(dec, f);   // dcmcore's modality-LUT (HU) float / display
                if (!fv) throw std::runtime_error(std::string("frame decode failed: ")
                                                  + std::string(rdr::describe(fv.error())));
                if (f == first) {
                    rows = fv->height; cols = fv->width;
                    const fmt_info fi = display_fmt(fv->pixel_format, fv->samples_per_pixel);
                    dt_code = fi.code; ba = fi.bits; spp = fi.samples; dtype_name = fi.name;
                    bs = ba;
                    out->reserve(fv->bytes.size() * frames);
                }
                out->insert(out->end(), fv->bytes.begin(), fv->bytes.end());
            } else {
                auto rf = rdr::extract_raw_frame(dec, f);  // native stored integers (lossless)
                if (!rf) throw std::runtime_error(std::string("frame decode failed: ")
                                                  + std::string(rdr::describe(rf.error())));
                if (f == first) {
                    rows = rf->height; cols = rf->width; spp = rf->samples_per_pixel;
                    bs = rf->bits_stored; planar = rf->planar_configuration; sgn = rf->is_signed;
                    const std::uint64_t px = std::uint64_t(rf->width) * rf->height * spp;
                    ba = px ? static_cast<std::uint32_t>(rf->bytes.size() / px) * 8 : 0;
                    dt_code = (std::uint8_t)(sgn ? nb::dlpack::dtype_code::Int
                                                 : nb::dlpack::dtype_code::UInt);
                    dtype_name = dtype_int(ba, sgn);
                    out->reserve(rf->bytes.size() * frames);
                }
                out->insert(out->end(), rf->bytes.begin(), rf->bytes.end());
            }
        }
    } catch (...) { delete out; throw; }
    rdr::decoded& dec = *dec_owner;   // GIL re-acquired — Python assembly below

    std::size_t shape[4]; std::size_t ndim = 3;
    shape[0] = frames; shape[1] = rows; shape[2] = cols;
    if (spp > 1) { shape[3] = spp; ndim = 4; }

    nb::capsule owner(out, [](void* p) noexcept {
        delete static_cast<std::vector<std::byte>*>(p);
    });
    nb::dlpack::dtype dt{dt_code, static_cast<std::uint8_t>(ba), 1};
    nb::ndarray<nb::numpy> arr(out->data(), ndim, shape, owner, nullptr, dt);

    nb::dict meta;
    meta["frames"]               = frames;
    meta["rows"]                 = rows;
    meta["columns"]              = cols;
    meta["samples_per_pixel"]    = spp;
    meta["bits_allocated"]       = ba;
    meta["bits_stored"]          = bs;
    meta["pixel_representation"] = sgn ? 1 : 0;
    meta["planar_configuration"] = planar;
    meta["dtype"]                = dtype_name;
    meta["rescaled"]             = rescale;
    meta["byte_order"]           = "little";
    meta["bytes"]                = out->size();
    nb::list ns; for (std::size_t i = 0; i < ndim; ++i) ns.append(shape[i]);
    meta["numpy_shape"] = ns;
    add_meta(meta, dec.get());

    return nb::make_tuple(arr, meta);
}

// decode(path, frame=0, rescale=False) -> (ndarray[frames,rows,cols(,samples)], meta dict)
nb::object decode(const std::string& path, int frame, bool rescale) {
    std::vector<std::byte> bytes;
    { nb::gil_scoped_release rel; bytes = slurp(path); }   // file IO off-GIL
    return decode_span(std::span<const std::byte>{bytes}, frame, rescale, path);
}

// decode_bytes(part10, frame=0, rescale=False) -> same, decoding an in-memory Part-10
// buffer (no file). Lets a from-scratch Dataset's pixel_array reuse the native engine.
nb::object decode_bytes(nb::bytes part10, int frame, bool rescale) {
    std::span<const std::byte> sp{reinterpret_cast<const std::byte*>(part10.c_str()),
                                  part10.size()};
    return decode_span(sp, frame, rescale, "<bytes>");
}

std::uint32_t n_frames(const std::string& path) {
    const auto bytes = slurp(path);
    rdr::decoded dec{std::span<const std::byte>{bytes}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    return dec.frame_count();
}

// Is this a DICOM file? Accept a known extension OR the PS3.10 "DICM" preamble
// (so extension-less clinical exports are found).
bool is_dicom_file(const std::filesystem::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower((unsigned char)c));
    if (ext == ".dcm" || ext == ".dicom" || ext == ".ima") return true;
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(128);
    char magic[4] = {};
    return f.read(magic, 4) && std::memcmp(magic, "DICM", 4) == 0;
}

// scan_dicom_dir(root, recursive) -> sorted list of DICOM file paths.
// Directory discovery in C++ (no Python-side logic); the glob-pattern variant
// stays in the thin Python layer.
std::vector<std::string> scan_dicom_dir(const std::string& root, bool recursive) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    std::error_code ec;
    fs::path r(root);
    if (fs::is_regular_file(r, ec)) { out.push_back(root); return out; }
    auto consider = [&](const fs::directory_entry& e) {
        if (e.is_regular_file(ec) && is_dicom_file(e.path())) out.push_back(e.path().string());
    };
    if (recursive)
        for (auto it = fs::recursive_directory_iterator(r, ec);
             !ec && it != fs::recursive_directory_iterator(); ++it) consider(*it);
    else
        for (auto it = fs::directory_iterator(r, ec);
             !ec && it != fs::directory_iterator(); ++it) consider(*it);
    std::sort(out.begin(), out.end());
    return out;
}

// assemble_volume(paths) -> (ndarray[depth, rows, cols] float32 HU, meta)
//
// Thin nanobind shim over dcmbase::volume::assemble (the shared engine wrapper
// that pydcm load_series AND the dcm2nii CLI both use). The geometry — IOP
// clustering, IPP-projection Z-sort, N-D dimension discovery, modality-LUT (HU)
// — all lives in that helper / the C engine; here we only wrap it as NumPy.
nb::object assemble_volume(const std::vector<std::string>& paths) {
    dcmbase::volume::assembled v = [&] {
        nb::gil_scoped_release rel;   // N-file decode + geometry: pure C++
        return dcmbase::volume::assemble(paths);
    }();

    const std::uint32_t bd = v.depth, br = v.rows, bc = v.cols;
    const std::size_t voxels = std::size_t(bc) * br * bd;
    auto* out = new std::vector<std::byte>(voxels * sizeof(float));
    std::memcpy(out->data(), v.data.data(), out->size());

    std::size_t shape[3] = {bd, br, bc};   // [D, H, W]
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(out->data(), 3, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});

    nb::dict meta;
    meta["depth"] = bd; meta["rows"] = br; meta["columns"] = bc;
    meta["dtype"] = "float32"; meta["rescaled"] = true; meta["series_instance_uid"] = v.series_instance_uid;
    nb::list sp; sp.append(v.spacing[2]); sp.append(v.spacing[1]); sp.append(v.spacing[0]);  // [slice,row,col]=[z,y,x]
    meta["spacing"] = sp;
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(v.affine[i]);
    meta["affine"] = aff;                          // voxel→world, column-major 4×4
    nb::list ns; ns.append(bd); ns.append(br); ns.append(bc);
    meta["numpy_shape"] = ns;
    return nb::make_tuple(arr, meta);
}

// assemble_4d(paths) -> (ndarray[V,depth,rows,cols] float32, meta). The non-Z
// axes (DWI direction / b-value / fMRI time) become the 4th dimension; meta
// carries the per-volume representative file + frame for the .bval/.bvec table.
nb::object assemble_4d(const std::vector<std::string>& paths) {
    dcmbase::volume::assembled_4d v = [&] {
        nb::gil_scoped_release rel;
        return dcmbase::volume::assemble_4d(paths);
    }();
    const std::uint32_t V = v.volumes, bd = v.depth, br = v.rows, bc = v.cols;
    const std::size_t total = std::size_t(V) * bd * br * bc;
    auto* out = new std::vector<std::byte>(total * sizeof(float));
    std::memcpy(out->data(), v.data.data(), out->size());

    std::size_t shape[4] = {V, bd, br, bc};
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(out->data(), 4, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});

    nb::dict meta;
    meta["volumes"] = V; meta["depth"] = bd; meta["rows"] = br; meta["columns"] = bc;
    meta["series_instance_uid"] = v.series_instance_uid;
    nb::list sp; sp.append(v.spacing[2]); sp.append(v.spacing[1]); sp.append(v.spacing[0]);
    meta["spacing"] = sp;
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(v.affine[i]);
    meta["affine"] = aff;
    nb::list vp; for (const auto& p : v.volume_path) vp.append(p);
    meta["volume_path"] = vp;
    nb::list vf; for (auto f : v.volume_frame) vf.append(f);
    meta["volume_frame"] = vf;
    // The varying non-spatial axes (slowest-first) that span the 4th dimension —
    // each {tag,name,values,spacing}. Empty list for a plain 3-D series.
    nb::list dims;
    for (const auto& ax : v.dims) {
        nb::dict d;
        d["tag"] = ax.tag; d["name"] = ax.name; d["spacing"] = ax.spacing;
        nb::list vals; for (float x : ax.values) vals.append(x);
        d["values"] = vals;
        dims.append(d);
    }
    meta["dimensions"] = dims;
    return nb::make_tuple(arr, meta);
}

// assemble_dwi(paths, order) -> (ndarray[V,Z,Y,X] float32, meta{bvals[V], bvecs[V][3]
// voxel-frame, affine, spacing, ...}). Groups single-frame files by CSA diffusion;
// the shared C++ engine pydcm load_dwi / dcm2nii / medfilm all build on. order:
// "gradient" (b0 first, deterministic) or "acquisition" (InstanceNumber order).
nb::object assemble_dwi(const std::vector<std::string>& paths, const std::string& order) {
    const bool acq = (order == "acquisition");
    dcmbase::volume::assembled_dwi v = [&] {
        nb::gil_scoped_release rel;
        return dcmbase::volume::assemble_dwi(paths, acq);
    }();
    const std::uint32_t V = v.volumes, bd = v.depth, br = v.rows, bc = v.cols;
    auto* out = new std::vector<std::byte>(std::size_t(V) * bd * br * bc * sizeof(float));
    std::memcpy(out->data(), v.data.data(), out->size());

    std::size_t shape[4] = {V, bd, br, bc};
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(out->data(), 4, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});

    nb::dict meta;
    meta["volumes"] = V;
    meta["series_instance_uid"] = v.series_instance_uid;
    nb::list sp; sp.append(v.spacing[2]); sp.append(v.spacing[1]); sp.append(v.spacing[0]);
    meta["spacing"] = sp;
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(v.affine[i]);
    meta["affine"] = aff;
    nb::list bv; for (float b : v.bvalue) bv.append(b);
    meta["bvals"] = bv;
    nb::list bvec; for (const auto& g : v.gradient) { nb::list e; e.append(g[0]); e.append(g[1]); e.append(g[2]); bvec.append(e); }
    meta["bvecs"] = bvec;
    return nb::make_tuple(arr, meta);
}

// write_nifti_volume(array[D,H,W] or [V,D,H,W], affine[16] column-major LPS, path)
// Thin wrapper over dcmbase's dcm_nifti_write_nd: the numpy data is the assembled,
// i-fastest (cols fastest) volume (4D adds a volume-slowest axis — DWI directions
// / fMRI time), and model_matrix is the LPS affine the engine flips to RAS.
// ".nii.gz" path → gzip. The NIfTI datatype follows the array dtype.
void write_nifti_volume(nb::ndarray<nb::c_contig> arr,
                        const std::vector<float>& affine, const std::string& path) {
    if (arr.ndim() != 3 && arr.ndim() != 4)
        throw std::runtime_error("write_nifti_volume: array must be 3D [D,H,W] or 4D [V,D,H,W]");
    if (affine.size() != 16) throw std::runtime_error("write_nifti_volume: affine must have 16 elements");

    using C = nb::dlpack::dtype_code;
    const auto d = arr.dtype();
    vol_pixel_fmt_t fmt;
    if      (d.code == (std::uint8_t)C::Float && d.bits == 32) fmt = VOL_FMT_FLOAT;
    else if (d.code == (std::uint8_t)C::UInt  && d.bits == 8)  fmt = VOL_FMT_UINT8;
    else if (d.code == (std::uint8_t)C::UInt  && d.bits == 16) fmt = VOL_FMT_UINT16;
    else if (d.code == (std::uint8_t)C::Int   && d.bits == 16) fmt = VOL_FMT_INT16;
    else throw std::runtime_error("write_nifti_volume: unsupported dtype (float32/uint8/int16/uint16)");

    std::uint32_t volumes = 1, depth, rows, cols;
    if (arr.ndim() == 4) {
        volumes = (std::uint32_t)arr.shape(0); depth = (std::uint32_t)arr.shape(1);
        rows    = (std::uint32_t)arr.shape(2); cols  = (std::uint32_t)arr.shape(3);
    } else {
        depth = (std::uint32_t)arr.shape(0); rows = (std::uint32_t)arr.shape(1); cols = (std::uint32_t)arr.shape(2);
    }
    float mm[16]; for (int i = 0; i < 16; ++i) mm[i] = affine[i];
    const int rc = dcm_nifti_write_nd(path.c_str(), arr.data(), cols, rows, depth, volumes, mm, fmt, 1.0f, 0.0f);
    if (rc != DCM_NIFTI_OK)
        throw std::runtime_error("dcm_nifti_write_nd failed: rc=" + std::to_string(rc));
}

// read_nifti(path) -> (ndarray[depth, rows, cols], meta{affine column-major LPS,
// spacing[z,y,x], dtype}). Reads .nii or .nii.gz; the file's RAS sform is
// flipped back to our LPS convention by the engine.
nb::object read_nifti(const std::string& path) {
    dcm_nifti_image_t img;
    const int rc = dcm_nifti_read(path.c_str(), &img);
    if (rc != DCM_NIFTI_OK)
        throw std::runtime_error("dcm_nifti_read failed: rc=" + std::to_string(rc));

    // NIfTI dims are X(cols),Y(rows),Z(depth); data is cols-fastest = C-order
    // for numpy shape [depth, rows, cols].
    const std::uint32_t nx = img.dim[0], ny = img.dim[1], nz = img.dim[2],
                        nt = img.dim[3] ? img.dim[3] : 1;
    const std::size_t voxels = std::size_t(nx) * ny * nz * nt * img.channels;  // incl. time (4D)
    const std::size_t bytes  = voxels * img.bytes_per_voxel;

    nb::dlpack::dtype dt;
    switch (img.datatype) {
        case DCM_NIFTI_DT_FLOAT32: dt = {(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1}; break;
        case DCM_NIFTI_DT_UINT8:   dt = {(std::uint8_t)nb::dlpack::dtype_code::UInt,  8,  1}; break;
        case DCM_NIFTI_DT_INT8:    dt = {(std::uint8_t)nb::dlpack::dtype_code::Int,   8,  1}; break;
        case DCM_NIFTI_DT_INT16:   dt = {(std::uint8_t)nb::dlpack::dtype_code::Int,   16, 1}; break;
        case DCM_NIFTI_DT_UINT16:  dt = {(std::uint8_t)nb::dlpack::dtype_code::UInt,  16, 1}; break;
        case DCM_NIFTI_DT_INT32:   dt = {(std::uint8_t)nb::dlpack::dtype_code::Int,   32, 1}; break;
        case DCM_NIFTI_DT_UINT32:  dt = {(std::uint8_t)nb::dlpack::dtype_code::UInt,  32, 1}; break;
        default: free(img.data); throw std::runtime_error("read_nifti: unsupported datatype");
    }

    auto* owned = new std::vector<std::byte>(bytes);
    std::memcpy(owned->data(), img.data, bytes);
    free(img.data);

    // [t, depth, rows, cols] for 4D, [depth, rows, cols] for 3D (cols-fastest = C-order).
    std::size_t shape4[4] = {nt, nz, ny, nx};
    const std::size_t ndim = (nt > 1) ? 4 : 3;
    std::size_t* shape = (nt > 1) ? shape4 : shape4 + 1;
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(owned->data(), ndim, shape, owner, nullptr, dt);

    nb::dict meta;
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(img.affine_lps[i]);
    meta["affine"] = aff;                          // voxel→world LPS, column-major
    nb::list sp; sp.append(img.pixdim[2]); sp.append(img.pixdim[1]); sp.append(img.pixdim[0]);  // [z,y,x]
    meta["spacing"] = sp;
    meta["time_points"] = nt;
    meta["scl_slope"] = img.scl_slope; meta["scl_inter"] = img.scl_inter;
    return nb::make_tuple(arr, meta);
}

// ── pydcm.transforms — thin marshalling over dcmbase::transform (the CPU "ITK").
// numpy [D,H,W] + LPS affine[16] col-major ⇄ tf::array (single channel). Data is
// W-fastest (C-contig) == tf::array X-fastest, so a straight memcpy suffices.
namespace {
namespace tf = dcmbase::transform;

vol_pixel_fmt_t tf_fmt(nb::dlpack::dtype d) {
    using C = nb::dlpack::dtype_code;
    if (d.code == (std::uint8_t)C::Float && d.bits == 32) return VOL_FMT_FLOAT;
    if (d.code == (std::uint8_t)C::UInt  && d.bits == 8)  return VOL_FMT_UINT8;
    if (d.code == (std::uint8_t)C::UInt  && d.bits == 16) return VOL_FMT_UINT16;
    if (d.code == (std::uint8_t)C::Int   && d.bits == 16) return VOL_FMT_INT16;
    throw std::runtime_error("transform: unsupported dtype (float32/uint8/int16/uint16)");
}

// True when the ndarray is float64 (numpy/torch default for many ops). The engine
// tops out at float32, so f64 input is accepted and down-cast on copy.
bool is_f64(nb::dlpack::dtype d) {
    return d.code == (std::uint8_t)nb::dlpack::dtype_code::Float && d.bits == 64;
}

tf::array tf_from_numpy(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine, bool is_label) {
    if (arr.ndim() != 3) throw std::runtime_error("transform: array must be 3D [D,H,W]");
    if (affine.size() != 16) throw std::runtime_error("transform: affine must have 16 elements");
    tf::array a;
    a.dims[0] = (std::uint32_t)arr.shape(2);   // cols (X) = W
    a.dims[1] = (std::uint32_t)arr.shape(1);   // rows (Y) = H
    a.dims[2] = (std::uint32_t)arr.shape(0);   // depth (Z) = D
    a.channels = 1; a.is_label = is_label;
    for (int i = 0; i < 16; ++i) a.model_matrix[i] = affine[i];
    if (is_f64(arr.dtype())) {                 // float64 → float32 down-cast
        a.fmt = VOL_FMT_FLOAT;
        const std::size_t n = a.voxels();
        a.data.resize(n * 4);
        const double* s = reinterpret_cast<const double*>(arr.data());
        float* dst = reinterpret_cast<float*>(a.data.data());
        for (std::size_t i = 0; i < n; ++i) dst[i] = float(s[i]);
    } else {
        a.fmt = tf_fmt(arr.dtype());
        a.data.resize(a.voxels() * a.bpv());
        std::memcpy(a.data.data(), arr.data(), a.data.size());
    }
    return a;
}

nb::dlpack::dtype tf_dtype(vol_pixel_fmt_t fmt) {
    using C = nb::dlpack::dtype_code;
    switch (fmt) {
        case VOL_FMT_UINT8:  return {(std::uint8_t)C::UInt,  8,  1};
        case VOL_FMT_UINT16: return {(std::uint8_t)C::UInt,  16, 1};
        case VOL_FMT_INT16:  return {(std::uint8_t)C::Int,   16, 1};
        default:             return {(std::uint8_t)C::Float, 32, 1};
    }
}

// tf::array → (ndarray[D,H,W], meta{affine}). Moves the bytes into a capsule-owned
// vector (zero-copy to numpy).
nb::object tf_to_numpy(tf::array&& a) {
    auto* owned = new std::vector<std::byte>(std::move(a.data));
    std::size_t shape[3] = { a.dims[2], a.dims[1], a.dims[0] };   // [D,H,W]
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(owned->data(), 3, shape, owner, nullptr, tf_dtype(a.fmt));
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(a.model_matrix[i]);
    nb::dict meta; meta["affine"] = aff;
    return nb::make_tuple(arr, meta);
}
}  // namespace

// transform_resample_to_spacing(arr[D,H,W], affine[16] LPS, spacing[x,y,z] mm,
// is_label, interp{"linear"|"cubic"|"nearest"}) -> (arr, meta{affine}). Labels
// force nearest (vol engine, no class blending).
nb::object transform_resample_to_spacing(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                         const std::vector<float>& spacing, bool is_label,
                                         const std::string& interp) {
    if (spacing.size() != 3) throw std::runtime_error("resample: spacing must be [x,y,z]");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    const float sp[3] = { spacing[0], spacing[1], spacing[2] };   // (col/x, row/y, slice/z)
    tf::interp iz = interp == "nearest" ? tf::interp::nearest
                  : interp == "cubic"   ? tf::interp::cubic : tf::interp::linear;
    return tf_to_numpy(tf::resample_to_spacing(src, sp, iz));
}

// transform_resample_to_reference(moving[D,H,W], moving_affine, ref_shape[D,H,W],
// ref_affine, is_label, interp) -> (arr on the reference grid, meta{ref affine}).
nb::object transform_resample_to_reference(nb::ndarray<nb::c_contig> moving, const std::vector<float>& moving_affine,
                                           const std::vector<std::uint32_t>& ref_shape,
                                           const std::vector<float>& ref_affine,
                                           bool is_label, const std::string& interp, double fill) {
    if (ref_shape.size() != 3) throw std::runtime_error("resample_to_reference: ref_shape must be [D,H,W]");
    if (ref_affine.size() != 16) throw std::runtime_error("resample_to_reference: ref_affine must have 16 elements");
    tf::array mov = tf_from_numpy(moving, moving_affine, is_label);
    tf::array ref;                                                 // grid only — no data needed
    ref.dims[0] = ref_shape[2]; ref.dims[1] = ref_shape[1]; ref.dims[2] = ref_shape[0];
    ref.fmt = mov.fmt; ref.channels = 1;
    for (int i = 0; i < 16; ++i) ref.model_matrix[i] = ref_affine[i];
    tf::interp iz = interp == "nearest" ? tf::interp::nearest
                  : interp == "cubic"   ? tf::interp::cubic : tf::interp::linear;
    return tf_to_numpy(tf::resample_to_reference(mov, ref, iz, float(fill)));
}

// transform_affine(arr[D,H,W], affine, matrix[16] col-major voxel→voxel, is_label, interp)
// -> (arr on the same grid, meta). matrix is the forward transform applied to the image.
nb::object transform_affine(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                            const std::vector<float>& matrix, bool is_label, const std::string& interp) {
    if (matrix.size() != 16) throw std::runtime_error("affine: matrix must have 16 elements (column-major)");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    float M[16]; for (int i = 0; i < 16; ++i) M[i] = matrix[i];
    tf::interp iz = interp == "nearest" ? tf::interp::nearest
                  : interp == "cubic"   ? tf::interp::cubic : tf::interp::linear;
    return tf_to_numpy(tf::affine(src, M, iz));
}

// transform_resample_separate_z(arr[D,H,W], affine, out_shape[D,H,W]) -> (arr, meta).
nb::object transform_resample_separate_z(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                         const std::vector<std::uint32_t>& out_shape) {
    if (out_shape.size() != 3) throw std::runtime_error("resample_separate_z: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    return tf_to_numpy(tf::resample_separate_z(src, od));
}

nb::object transform_resample_cubic(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                    const std::vector<std::uint32_t>& out_shape) {
    if (out_shape.size() != 3) throw std::runtime_error("resample_cubic: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    return tf_to_numpy(tf::resample_cubic(src, od));
}

// transform_pil_resize2d(arr[D,H,W], affine, out_h, out_w, filter) -> (arr[D,out_h,out_w], meta).
nb::object transform_pil_resize2d(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                  std::uint32_t out_h, std::uint32_t out_w, const std::string& filter) {
    tf::array src = tf_from_numpy(arr, affine, false);
    const tf::pil_filter f = filter == "bilinear" ? tf::pil_filter::bilinear
                           : filter == "bicubic"  ? tf::pil_filter::bicubic
                           : throw std::runtime_error("pil_resize2d: filter must be 'bicubic' or 'bilinear'");
    return tf_to_numpy(tf::pil_resize2d(src, out_h, out_w, f));
}

// transform_bilinear_resize2d(arr[D,H,W], affine, out_h, out_w) -> (arr[D,out_h,out_w], meta).
nb::object transform_bilinear_resize2d(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                       std::uint32_t out_h, std::uint32_t out_w) {
    tf::array src = tf_from_numpy(arr, affine, false);
    return tf_to_numpy(tf::bilinear_resize2d(src, out_h, out_w));
}

nb::object transform_resample_nearest(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                      const std::vector<std::uint32_t>& out_shape) {
    if (out_shape.size() != 3) throw std::runtime_error("resample_nearest: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, true);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    return tf_to_numpy(tf::resample_nearest(src, od));
}

nb::object transform_resample_grid_sample(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                          const std::vector<std::uint32_t>& out_shape) {
    if (out_shape.size() != 3) throw std::runtime_error("resample_grid_sample: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    return tf_to_numpy(tf::resample_grid_sample(src, od));
}

nb::object transform_resample(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                              const std::vector<std::uint32_t>& out_shape, const std::string& backend) {
    if (out_shape.size() != 3) throw std::runtime_error("resample: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    const tf::convention conv = backend == "skimage" ? tf::convention::skimage
                              : backend == "torch"   ? tf::convention::torch
                              : backend == "itk"     ? tf::convention::itk
                              : throw std::runtime_error("resample: backend must be 'skimage', 'torch', or 'itk'");
    return tf_to_numpy(tf::resample(src, od, conv));
}

// transform_normalize_zscore(arr[D,H,W], affine, nonzero) -> (arr float32, meta).
nb::object transform_normalize_zscore(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine, bool nonzero) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::normalize_zscore(a, nonzero);
    return tf_to_numpy(std::move(a));
}

// transform_scale_intensity_range(arr, affine, a0,a1,b0,b1, clip) -> (arr float32, meta).
nb::object transform_scale_intensity_range(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                           float a0, float a1, float b0, float b1, bool clip) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::scale_intensity_range(a, a0, a1, b0, b1, clip);
    return tf_to_numpy(std::move(a));
}

// transform_normalize_ct(arr, affine, clip_lo,clip_hi, mean,std) -> (arr float32, meta).
nb::object transform_normalize_ct(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                  float clip_lo, float clip_hi, float mean, float stddev) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::normalize_ct(a, clip_lo, clip_hi, mean, stddev);
    return tf_to_numpy(std::move(a));
}

// transform_rescale_robust(arr, affine, dst_min,dst_max, f_low,f_high) -> (arr float32, meta).
nb::object transform_rescale_robust(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                    double dst_min, double dst_max, double f_low, double f_high) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::rescale_robust(a, dst_min, dst_max, f_low, f_high);
    return tf_to_numpy(std::move(a));
}

// transform_scale_intensity_range_percentiles(arr, affine, lower,upper, b0,b1, clip) -> (arr, meta).
nb::object transform_scale_intensity_range_percentiles(nb::ndarray<nb::c_contig> arr,
        const std::vector<float>& affine, float lower, float upper, float b0, float b1, bool clip) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::scale_intensity_range_percentiles(a, lower, upper, b0, b1, clip);
    return tf_to_numpy(std::move(a));
}

// transform_adjust_contrast(arr, affine, gamma) -> (arr float32, meta).
nb::object transform_adjust_contrast(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine, float gamma) {
    tf::array a = tf_from_numpy(arr, affine, false);
    tf::adjust_contrast(a, gamma);
    return tf_to_numpy(std::move(a));
}

// transform_gaussian_smooth(arr[D,H,W], affine, sigma[z,y,x] voxels) -> (arr float32, meta).
nb::object transform_gaussian_smooth(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                     const std::vector<float>& sigma) {
    if (sigma.size() != 3) throw std::runtime_error("gaussian_smooth: sigma must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const float sg[3] = { sigma[2], sigma[1], sigma[0] };          // [z,y,x] → engine [x,y,z]
    return tf_to_numpy(tf::gaussian_smooth(src, sg));
}

// transform_argmax(probs[D,H,W,C], affine) -> (labels[D,H,W] uint8/16, meta). C
// (classes) is the fastest axis (channel-last, C-contig).
nb::object transform_argmax(nb::ndarray<nb::c_contig> probs, const std::vector<float>& affine) {
    if (probs.ndim() != 4) throw std::runtime_error("argmax: probs must be 4D [D,H,W,C]");
    if (affine.size() != 16) throw std::runtime_error("argmax: affine must have 16 elements");
    tf::array p;
    p.dims[0] = (std::uint32_t)probs.shape(2);   // cols
    p.dims[1] = (std::uint32_t)probs.shape(1);   // rows
    p.dims[2] = (std::uint32_t)probs.shape(0);   // depth
    p.channels = (std::uint32_t)probs.shape(3);
    for (int i = 0; i < 16; ++i) p.model_matrix[i] = affine[i];
    const std::size_t ne = p.voxels() * p.channels;
    if (is_f64(probs.dtype())) {                 // float64 → float32 down-cast
        p.fmt = VOL_FMT_FLOAT;
        p.data.resize(ne * 4);
        const double* s = reinterpret_cast<const double*>(probs.data());
        float* dst = reinterpret_cast<float*>(p.data.data());
        for (std::size_t i = 0; i < ne; ++i) dst[i] = float(s[i]);
    } else {
        p.fmt = tf_fmt(probs.dtype());
        p.data.resize(ne * p.bpv());
        std::memcpy(p.data.data(), probs.data(), p.data.size());
    }
    return tf_to_numpy(tf::argmax(p));
}

// P2 spatial ops. Shape/index tuples are in numpy [z,y,x] order (matching
// pixels.shape); converted to the engine's [x,y,z] here.
nb::object transform_resize(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                            const std::vector<std::uint32_t>& out_shape, bool is_label,
                            const std::string& interp) {
    if (out_shape.size() != 3) throw std::runtime_error("resize: out_shape must be [D,H,W]");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    const std::uint32_t od[3] = { out_shape[2], out_shape[1], out_shape[0] };
    tf::interp iz = interp == "nearest" ? tf::interp::nearest
                  : interp == "cubic"   ? tf::interp::cubic : tf::interp::linear;
    return tf_to_numpy(tf::resize(src, od, iz));
}

nb::object transform_crop(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                          const std::vector<std::uint32_t>& start, const std::vector<std::uint32_t>& size) {
    if (start.size() != 3 || size.size() != 3) throw std::runtime_error("crop: start/size must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t st[3] = { start[2], start[1], start[0] };
    const std::uint32_t sz[3] = { size[2],  size[1],  size[0]  };
    return tf_to_numpy(tf::crop(src, st, sz));
}

nb::object transform_pad(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                         const std::vector<std::uint32_t>& lo, const std::vector<std::uint32_t>& hi,
                         const std::string& mode, float value) {
    if (lo.size() != 3 || hi.size() != 3) throw std::runtime_error("pad: lo/hi must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, false);
    const std::uint32_t l[3] = { lo[2], lo[1], lo[0] }, h[3] = { hi[2], hi[1], hi[0] };
    tf::pad_mode m = mode == "edge" ? tf::pad_mode::edge
                   : mode == "reflect" ? tf::pad_mode::reflect : tf::pad_mode::constant;
    return tf_to_numpy(tf::pad(src, l, h, m, value));
}

nb::object transform_crop_foreground(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                     std::uint32_t margin) {
    tf::array src = tf_from_numpy(arr, affine, false);
    return tf_to_numpy(tf::crop_foreground(src, margin));
}

nb::object transform_center_crop(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                 const std::vector<std::uint32_t>& size, bool is_label) {
    if (size.size() != 3) throw std::runtime_error("center_crop: size must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    const std::uint32_t sz[3] = { size[2], size[1], size[0] };
    return tf_to_numpy(tf::center_crop(src, sz));
}

nb::object transform_spatial_pad(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                 const std::vector<std::uint32_t>& size, const std::string& mode,
                                 float value, bool is_label) {
    if (size.size() != 3) throw std::runtime_error("spatial_pad: size must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    const std::uint32_t sz[3] = { size[2], size[1], size[0] };
    tf::pad_mode m = mode == "edge" ? tf::pad_mode::edge
                   : mode == "reflect" ? tf::pad_mode::reflect : tf::pad_mode::constant;
    return tf_to_numpy(tf::spatial_pad(src, sz, m, value));
}

nb::object transform_divisible_pad(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                   const std::vector<std::uint32_t>& k, const std::string& mode,
                                   float value, bool is_label) {
    if (k.size() != 3) throw std::runtime_error("divisible_pad: k must be [z,y,x]");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    const std::uint32_t kk[3] = { k[2], k[1], k[0] };
    tf::pad_mode m = mode == "edge" ? tf::pad_mode::edge
                   : mode == "reflect" ? tf::pad_mode::reflect : tf::pad_mode::constant;
    return tf_to_numpy(tf::divisible_pad(src, kk, m, value));
}

// rotate90(arr, affine, k, axis0, axis1) — axes are numpy [D,H,W] axis indices (0,1,2),
// mapped to engine [x,y,z] as 2-axis. is_label preserved through dtype.
nb::object transform_rotate90(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                              int k, int axis0, int axis1, bool is_label) {
    if (axis0 < 0 || axis0 > 2 || axis1 < 0 || axis1 > 2 || axis0 == axis1)
        throw std::runtime_error("rotate90: axes must be two distinct of {0,1,2}");
    tf::array src = tf_from_numpy(arr, affine, is_label);
    return tf_to_numpy(tf::rotate90(src, k, 2 - axis0, 2 - axis1));
}

// sliding_window_positions(spatial[z,y,x], roi[z,y,x], overlap) -> flat [z,y,x] origins
// (3 per patch). Inputs/outputs in numpy axis order.
std::vector<std::uint32_t> transform_sliding_window_positions(
        const std::vector<std::uint32_t>& spatial, const std::vector<std::uint32_t>& roi, double overlap) {
    if (spatial.size() != 3 || roi.size() != 3) throw std::runtime_error("sliding_window_positions: spatial/roi must be [z,y,x]");
    const std::uint32_t sp[3] = { spatial[2], spatial[1], spatial[0] };
    const std::uint32_t rr[3] = { roi[2], roi[1], roi[0] };
    const std::vector<std::uint32_t> xyz = tf::sliding_window_positions(sp, rr, overlap);
    std::vector<std::uint32_t> zyx; zyx.reserve(xyz.size());
    for (std::size_t i = 0; i < xyz.size(); i += 3) {             // (x,y,z) -> (z,y,x)
        zyx.push_back(xyz[i + 2]); zyx.push_back(xyz[i + 1]); zyx.push_back(xyz[i]);
    }
    return zyx;
}

// gaussian_importance_map(roi[z,y,x], sigma_scale, convention) -> (map[D,H,W] float32, meta).
nb::object transform_gaussian_importance_map(const std::vector<std::uint32_t>& roi, double sigma_scale,
                                             const std::string& convention) {
    if (roi.size() != 3) throw std::runtime_error("gaussian_importance_map: roi must be [z,y,x]");
    tf::gaussian_conv conv = convention == "monai" ? tf::gaussian_conv::monai
                           : convention == "nnunet" ? tf::gaussian_conv::nnunet
                           : throw std::runtime_error("gaussian_importance_map: convention must be 'nnunet' or 'monai'");
    const std::uint32_t rr[3] = { roi[2], roi[1], roi[0] };
    return tf_to_numpy(tf::gaussian_importance_map(rr, sigma_scale, conv));
}

nb::object transform_flip(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                          const std::vector<int>& axis) {
    if (axis.size() != 3) throw std::runtime_error("flip: axis must be [z,y,x] bools");
    tf::array src = tf_from_numpy(arr, affine, false);
    const bool ax[3] = { axis[2] != 0, axis[1] != 0, axis[0] != 0 };
    return tf_to_numpy(tf::flip(src, ax));
}

nb::object transform_transpose(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                               const std::vector<int>& axes) {
    if (axes.size() != 3) throw std::runtime_error("transpose: axes must be a 3-permutation in [z,y,x] order");
    tf::array src = tf_from_numpy(arr, affine, false);
    // pydcm passes a (z,y,x)-order permutation; the engine array is (x,y,z): eng[s] = 2 - axes[2-s].
    const int eng[3] = { 2 - axes[2], 2 - axes[1], 2 - axes[0] };
    return tf_to_numpy(tf::transpose(src, eng));
}

nb::object transform_reorient(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                              const std::string& axcodes) {
    if (axcodes.size() != 3) throw std::runtime_error("reorient: axcodes must be 3 letters (e.g. 'LPS')");
    tf::array src = tf_from_numpy(arr, affine, false);
    const char ax[3] = { axcodes[0], axcodes[1], axcodes[2] };
    return tf_to_numpy(tf::reorient(src, ax));
}

nb::object transform_connected_components(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                          int connectivity) {
    tf::array src = tf_from_numpy(arr, affine, true);
    return tf_to_numpy(tf::connected_components(src, connectivity));
}

nb::object transform_keep_largest_cc(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                     int connectivity, bool per_class) {
    tf::array src = tf_from_numpy(arr, affine, true);
    return tf_to_numpy(tf::keep_largest_connected_component(src, connectivity, per_class));
}

nb::object transform_fill_holes(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                int connectivity) {
    tf::array src = tf_from_numpy(arr, affine, true);
    return tf_to_numpy(tf::fill_holes(src, connectivity));
}

nb::object transform_as_discrete(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                 float threshold) {
    tf::array src = tf_from_numpy(arr, affine, false);
    return tf_to_numpy(tf::as_discrete(src, threshold));
}

nb::object transform_remove_small_objects(nb::ndarray<nb::c_contig> arr, const std::vector<float>& affine,
                                          int min_size, int connectivity, bool per_class) {
    tf::array src = tf_from_numpy(arr, affine, true);
    return tf_to_numpy(tf::remove_small_objects(src, min_size, connectivity, per_class));
}

// build_seg_from_nifti(ref_paths, mask_nii, segments) -> Part-10 SEG bytes.
// The NIfTI/FSL -> DICOM-SEG return path: reads the co-framed label volume,
// reorders Z to the reference's ascending-position order (affine-aware flip),
// and emits a coded Segmentation via the shared dcmbase::seg engine. `segments`
// is a list of dicts: {labelID, label, rgb:[r,g,b], category/type/anatomic:
// [value,scheme,meaning], algorithm_type, algorithm_name}.
nb::bytes build_seg_from_nifti(const std::vector<std::string>& ref_paths,
                               const std::string& mask_nii_path, nb::list segments) {
    auto geom = dcmbase::seg::reference_series(ref_paths);
    if (!geom) throw std::runtime_error("seg_from_nifti: no decodable reference-series slices");
    auto raw = dcmbase::seg::nifti_labelmap(mask_nii_path, geom);
    if (raw.empty())
        throw std::runtime_error("seg_from_nifti: NIfTI mask is not co-framed with the reference "
                                 "series (dimension mismatch or unreadable)");

    const auto get_str = [](const nb::dict& d, const char* k) -> std::string {
        if (d.contains(k)) { nb::object v = d[k]; if (!v.is_none()) return nb::cast<std::string>(v); }
        return {};
    };
    const auto get_code = [](const nb::dict& d, const char* k) -> dcmbase::seg::code {
        dcmbase::seg::code c;
        if (d.contains(k)) { nb::object v = d[k];
            if (!v.is_none()) { auto t = nb::cast<std::vector<std::string>>(v);
                if (t.size() >= 3) { c.value = t[0]; c.scheme = t[1]; c.meaning = t[2]; } } }
        return c;
    };

    std::vector<dcmbase::seg::segment> segs;
    std::vector<std::uint16_t> label_ids;
    std::uint16_t auto_id = 0;
    for (nb::handle h : segments) {
        nb::dict d = nb::cast<nb::dict>(h);
        dcmbase::seg::segment s;
        s.label          = get_str(d, "label");
        s.category       = get_code(d, "category");
        s.type           = get_code(d, "type");
        s.anatomic       = get_code(d, "anatomic");
        s.algorithm_type = get_str(d, "algorithm_type");
        s.algorithm_name = get_str(d, "algorithm_name");
        if (d.contains("rgb")) { nb::object v = d["rgb"];
            if (!v.is_none()) { auto c = nb::cast<std::vector<int>>(v);
                if (c.size() >= 3) { s.r = (std::uint8_t)c[0]; s.g = (std::uint8_t)c[1]; s.b = (std::uint8_t)c[2]; } } }
        std::uint16_t lid = 0;
        if (d.contains("labelID")) { nb::object v = d["labelID"]; if (!v.is_none()) lid = (std::uint16_t)nb::cast<int>(v); }
        if (lid == 0) lid = ++auto_id; else auto_id = std::max(auto_id, lid);
        segs.push_back(std::move(s));
        label_ids.push_back(lid);
    }
    if (segs.empty()) throw std::runtime_error("seg_from_nifti: no segments supplied");

    // Remap raw labelID values -> segment index+1, per ascending-position slice.
    const std::size_t npx = std::size_t(geom.cols) * geom.rows;
    const std::size_t nsl = geom.slices.size();
    const auto seg_for = [&](std::uint16_t v) -> int {
        if (!v) return -1;
        for (std::size_t k = 0; k < label_ids.size(); ++k) if (label_ids[k] == v) return (int)k;
        return -1;
    };
    std::vector<std::vector<std::uint16_t>> remapped(nsl);
    for (std::size_t s = 0; s < nsl; ++s) {
        remapped[s].assign(npx, 0);
        const std::uint16_t* src = raw.data() + s * npx;
        for (std::size_t i = 0; i < npx; ++i)
            if (int k = seg_for(src[i]); k >= 0) remapped[s][i] = (std::uint16_t)(k + 1);
        geom.slices[s].labelmap = remapped[s].data();
    }

    auto out = dcmbase::seg::build(geom.m, segs, geom.slices);
    if (out.empty())
        throw std::runtime_error("seg_from_nifti: seg::build produced no output (no segmented frames?)");
    return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

// mosaic_expand(part10_bytes) -> (ndarray[N, rows, cols], meta) | None.
// Siemens mosaic -> N de-tiled slices via the CSA header. None if not a mosaic.
nb::object mosaic_expand(nb::bytes data) {
    const std::span<const std::byte> fb{reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    dcmbase::mosaic::expanded ex = dcmbase::mosaic::expand(fb);
    if (!ex) return nb::none();

    const std::uint32_t N = (std::uint32_t)ex.slices.size();
    const std::size_t tile_bytes = (std::size_t)ex.rows * ex.cols * ex.bytes_per_px;
    auto* owned = new std::vector<std::byte>((std::size_t)N * tile_bytes);
    for (std::uint32_t k = 0; k < N; k++)
        std::memcpy(owned->data() + (std::size_t)k * tile_bytes, ex.slices[k].data(), tile_bytes);

    using C = nb::dlpack::dtype_code;
    const auto code = (std::uint8_t)(ex.is_signed ? C::Int : C::UInt);
    nb::dlpack::dtype dt;
    if      (ex.bytes_per_px == 1) dt = {code, 8,  1};
    else if (ex.bytes_per_px == 2) dt = {code, 16, 1};
    else if (ex.bytes_per_px == 4) dt = {code, 32, 1};
    else { delete owned; throw std::runtime_error("mosaic_expand: unsupported pixel size"); }

    std::size_t shape[3] = {N, ex.rows, ex.cols};
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(owned->data(), 3, shape, owner, nullptr, dt);

    nb::dict meta;
    nb::list ipp;
    for (const auto& p : ex.ipp) { nb::list a; a.append(p[0]); a.append(p[1]); a.append(p[2]); ipp.append(a); }
    meta["image_position_patient"] = ipp;
    nb::list iop; for (int i = 0; i < 6; i++) iop.append(ex.iop[i]);
    meta["image_orientation_patient"] = iop;
    nb::list nrm; for (int i = 0; i < 3; i++) nrm.append(ex.normal[i]);
    meta["slice_normal"] = nrm;
    nb::list sp; sp.append(ex.slice_spacing); sp.append(ex.spacing_row); sp.append(ex.spacing_col);
    meta["spacing"] = sp;                          // [z, y, x] mm
    meta["slices"] = N;
    return nb::make_tuple(arr, meta);
}

// siemens_diffusion(part10_bytes) -> {b_value, gradient:[x,y,z]} | None.
// Siemens DWI/DTI b-value + gradient direction from the CSA header (the .bval/
// .bvec source); None if the file carries no CSA diffusion data.
nb::object siemens_diffusion(nb::bytes data) {
    const std::span<const std::byte> fb{reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    const dcmbase::mosaic::diffusion_info d = dcmbase::mosaic::diffusion(fb);
    if (!d.present) return nb::none();
    nb::dict r;
    r["b_value"] = d.b_value;
    nb::list g; g.append(d.gradient[0]); g.append(d.gradient[1]); g.append(d.gradient[2]);
    r["gradient"] = g;
    return r;
}

// read_diffusion(part10_bytes) -> [{b_value, gradient:[x,y,z]}, ...] | None.
// Unified per-frame DWI: the STANDARD MR Diffusion sequence (0018,9087/9089,
// already parsed into dicom_info — the modern enhanced-MF path) first, then the
// legacy Siemens CSA fallback. The .bval/.bvec source for diffusion_table.
nb::object read_diffusion(nb::bytes data) {
    const std::span<const std::byte> fb{reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    nb::list out;

    rdr::decoded dec{fb};
    if (dec) {
        const dicom_info_t* info = dec.get();
        const std::uint32_t nf = dec.frame_count();
        if (info && info->per_frame_diffusion_bvalue) {       // standard enhanced-MF
            for (std::uint32_t f = 0; f < nf; ++f) {
                const float bv = info->per_frame_diffusion_bvalue[f];
                if (bv < 0.0f) continue;                      // -1 sentinel: not set this frame
                nb::dict e; e["b_value"] = (double)bv;
                nb::list g;
                if (info->per_frame_diffusion_direction) {
                    g.append(info->per_frame_diffusion_direction[f*3 + 0]);
                    g.append(info->per_frame_diffusion_direction[f*3 + 1]);
                    g.append(info->per_frame_diffusion_direction[f*3 + 2]);
                } else { g.append(0.0); g.append(0.0); g.append(0.0); }
                e["gradient"] = g;
                out.append(e);
            }
        }
    }
    if (out.size() == 0) {                                    // legacy Siemens CSA fallback
        const dcmbase::mosaic::diffusion_info d = dcmbase::mosaic::diffusion(fb);
        if (d.present) {
            nb::dict e; e["b_value"] = d.b_value;
            nb::list g; g.append(d.gradient[0]); g.append(d.gradient[1]); g.append(d.gradient[2]);
            e["gradient"] = g;
            out.append(e);
        }
    }
    if (out.size() == 0) return nb::none();
    return out;
}

// bids_sidecar(part10_bytes) -> BIDS JSON string (timing/sequence/geometry). Empty-object "{}\n" if the file has no readable dataset.
nb::object bids_sidecar(nb::bytes data) {
    const std::span<const std::byte> fb{reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    const std::string json = dcmbase::bids::to_json(dcmbase::bids::extract(fb));
    return nb::cast(json);
}

// dti_fit_maps(b0, dwi, bvals, bvecs, maps) -> {name: ndarray}.
// b0:[n_voxels]; dwi:[n_dirs, n_voxels] direction-major; bvals:[n_dirs];
// bvecs:[n_dirs, 3]. Runs the native dcm_dti pipeline (OLS fit -> eigen -> map).
// FA/MD/AD/RD/CL/CP/CS come back float [n_voxels]; DEC comes back uint8 [n_voxels, 4].
nb::dict dti_fit_maps(nb::ndarray<const float, nb::ndim<1>, nb::c_contig> b0,
                      nb::ndarray<const float, nb::ndim<2>, nb::c_contig> dwi,
                      nb::ndarray<const float, nb::ndim<1>, nb::c_contig> bvals,
                      nb::ndarray<const float, nb::ndim<2>, nb::c_contig> bvecs,
                      const std::vector<std::string>& maps, bool wls) {
    const std::uint32_t n_voxels = (std::uint32_t)b0.shape(0);
    const std::uint32_t n_dirs   = (std::uint32_t)bvals.shape(0);
    if (dwi.shape(0) != n_dirs || dwi.shape(1) != n_voxels)
        throw std::runtime_error("dti_fit_maps: dwi must be [n_dirs, n_voxels]");
    if (bvecs.shape(0) != n_dirs || bvecs.shape(1) != 3)
        throw std::runtime_error("dti_fit_maps: bvecs must be [n_dirs, 3]");

    dti_context_t ctx;
    if (dti_prepare(&ctx, bvals.data(), bvecs.data(), n_dirs) != 0)
        throw std::runtime_error("dti_fit_maps: dti_prepare failed (need 6..256 DW directions)");

    std::vector<float> tensors((std::size_t)n_voxels * 6);
    (wls ? dti_fit_wls : dti_fit)(&ctx, b0.data(), dwi.data(), tensors.data(), n_voxels);
    std::vector<float> evals((std::size_t)n_voxels * 3), evecs((std::size_t)n_voxels * 9);
    dti_eigen_batch(tensors.data(), evals.data(), evecs.data(), n_voxels);

    const auto type_of = [](const std::string& s) -> int {
        if (s == "FA") return DTI_MAP_FA;  if (s == "MD") return DTI_MAP_MD;
        if (s == "AD") return DTI_MAP_AD;  if (s == "RD") return DTI_MAP_RD;
        if (s == "DEC") return DTI_MAP_DEC; if (s == "CL") return DTI_MAP_CL;
        if (s == "CP") return DTI_MAP_CP;  if (s == "CS") return DTI_MAP_CS;
        return -1;
    };
    // Trace-normalized Westin (dipy convention), computed from eigenvalues here so
    // BOTH conventions are available: CL/CP/CS use the native Westin-1997 ÷λ1 form,
    // linearity/planarity/sphericity use ÷Σλ (matches dipy / Westin 2002).
    const auto westin_norm = [&](const std::string& s, float* buf) -> bool {
        const int w = (s == "linearity") ? 0 : (s == "planarity") ? 1 : (s == "sphericity") ? 2 : -1;
        if (w < 0) return false;
        for (std::uint32_t v = 0; v < n_voxels; ++v) {
            const float l1 = evals[v*3], l2 = evals[v*3+1], l3 = evals[v*3+2], sum = l1+l2+l3;
            const float r = sum > 1e-20f ? 1.0f / sum : 0.0f;
            buf[v] = (w == 0) ? (l1-l2)*r : (w == 1) ? 2.0f*(l2-l3)*r : 3.0f*l3*r;
        }
        return true;
    };

    nb::dict result;
    for (const auto& name : maps) {
        {   // dipy-convention Westin?
            auto* owned = new std::vector<float>(n_voxels);
            if (westin_norm(name, owned->data())) {
                std::size_t shape[1] = {n_voxels};
                nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
                result[name.c_str()] = nb::ndarray<nb::numpy>(owned->data(), 1, shape, owner, nullptr,
                    nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});
                continue;
            }
            delete owned;
        }
        const int t = type_of(name);
        if (t < 0) throw std::runtime_error("dti_fit_maps: unknown map '" + name + "'");
        if (t == DTI_MAP_DEC) {
            auto* owned = new std::vector<std::byte>((std::size_t)n_voxels * 4);
            dti_compute_map(DTI_MAP_DEC, evals.data(), evecs.data(), owned->data(), n_voxels);
            std::size_t shape[2] = {n_voxels, 4};
            nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
            result[name.c_str()] = nb::ndarray<nb::numpy>(owned->data(), 2, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        } else {
            auto* owned = new std::vector<float>(n_voxels);
            dti_compute_map((dti_map_type_t)t, evals.data(), evecs.data(), owned->data(), n_voxels);
            std::size_t shape[1] = {n_voxels};
            nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
            result[name.c_str()] = nb::ndarray<nb::numpy>(owned->data(), 1, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});
        }
    }
    return result;
}

// dti_eigen(b0, dwi, bvals, bvecs, wls) -> (evals[n_voxels,3], evecs[n_voxels,9]).
// Raw tensor estimation output (sorted eigenvalues λ1≥λ2≥λ3 + column-major
// eigenvectors v1,v2,v3) for validating against a reference DTI implementation.
nb::object dti_eigen(nb::ndarray<const float, nb::ndim<1>, nb::c_contig> b0,
                     nb::ndarray<const float, nb::ndim<2>, nb::c_contig> dwi,
                     nb::ndarray<const float, nb::ndim<1>, nb::c_contig> bvals,
                     nb::ndarray<const float, nb::ndim<2>, nb::c_contig> bvecs, bool wls) {
    const std::uint32_t n_voxels = (std::uint32_t)b0.shape(0);
    const std::uint32_t n_dirs   = (std::uint32_t)bvals.shape(0);
    if (dwi.shape(0) != n_dirs || dwi.shape(1) != n_voxels || bvecs.shape(0) != n_dirs || bvecs.shape(1) != 3)
        throw std::runtime_error("dti_eigen: shape mismatch (dwi [n_dirs,n_voxels], bvecs [n_dirs,3])");

    dti_context_t ctx;
    if (dti_prepare(&ctx, bvals.data(), bvecs.data(), n_dirs) != 0)
        throw std::runtime_error("dti_eigen: dti_prepare failed (need 6..256 DW directions)");

    std::vector<float> tensors((std::size_t)n_voxels * 6);
    (wls ? dti_fit_wls : dti_fit)(&ctx, b0.data(), dwi.data(), tensors.data(), n_voxels);

    auto* va = new std::vector<float>((std::size_t)n_voxels * 3);
    auto* ve = new std::vector<float>((std::size_t)n_voxels * 9);
    dti_eigen_batch(tensors.data(), va->data(), ve->data(), n_voxels);

    std::size_t vs[2] = {n_voxels, 3}, es[2] = {n_voxels, 9};
    nb::capsule oa(va, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    nb::capsule oe(ve, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    const nb::dlpack::dtype f32{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
    return nb::make_tuple(nb::ndarray<nb::numpy>(va->data(), 2, vs, oa, nullptr, f32),
                          nb::ndarray<nb::numpy>(ve->data(), 2, es, oe, nullptr, f32));
}

// dti_track(evecs[nv,9], fa[nv], cols, rows, depth, spacing[3], ...) -> list of
// [P,3] streamlines in MILLIMETRE coordinates (voxel position x spacing — the
// physical coords dcmrender's dcm_fiber renders for medfilm). Wraps the native
// dcm_dti deterministic RK4 tracker. Voxel order is z*rows*cols + y*cols + x
// (flattened [depth,rows,cols]); divide a streamline point by `spacing` for voxels.
nb::list dti_track_streamlines(
        nb::ndarray<const float, nb::ndim<2>, nb::c_contig> evecs,
        nb::ndarray<const float, nb::ndim<1>, nb::c_contig> fa,
        std::uint32_t cols, std::uint32_t rows, std::uint32_t depth,
        nb::ndarray<const float, nb::ndim<1>, nb::c_contig> spacing,
        float fa_threshold, float angle_threshold, float step_size,
        std::uint32_t max_steps, float seed_fa_min,
        std::uint32_t max_tracks, std::uint32_t max_total_points) {
    const std::uint32_t nv = cols * rows * depth;
    if (evecs.shape(0) != nv || evecs.shape(1) != 9 || fa.shape(0) != nv || spacing.shape(0) != 3)
        throw std::runtime_error("dti_track: evecs [nv,9] / fa [nv] / spacing [3] must match cols*rows*depth");

    dti_track_params_t p{fa_threshold, angle_threshold, step_size, max_steps, seed_fa_min};
    std::vector<float> pts((std::size_t)max_total_points * 3);
    std::vector<std::uint32_t> offs(max_tracks), cnts(max_tracks);
    const std::uint32_t n = dti_track(evecs.data(), fa.data(), cols, rows, depth, spacing.data(), &p,
                                      pts.data(), offs.data(), cnts.data(), max_tracks, max_total_points);
    nb::list out;
    for (std::uint32_t t = 0; t < n; ++t) {
        const std::uint32_t off = offs[t], cnt = cnts[t];
        auto* owned = new std::vector<float>((std::size_t)cnt * 3);
        std::memcpy(owned->data(), &pts[(std::size_t)off * 3], (std::size_t)cnt * 3 * sizeof(float));
        std::size_t shape[2] = {cnt, 3};
        nb::capsule owner(owned, [](void* q) noexcept { delete static_cast<std::vector<float>*>(q); });
        out.append(nb::ndarray<nb::numpy>(owned->data(), 2, shape, owner, nullptr,
            nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1}));
    }
    return out;
}

// read_json(path) -> DICOM JSON Model string (every element, charset → UTF-8).
// Reuses dcmcorexx::bridge::to_dicom_json — the same converter dcm2json uses.
// UN-VR resolver backed by the native union dictionary (dcmbasexx::dict, 17,699
// entries — far richer than dcmcore's built-in C dict). Returns the libdcm VR
// code for a standard tag's known, unambiguous VR; 0 for unknown/private or an
// ambiguous spec VR ("OB or OW"). Implements the UN→known-VR replacement that drives so
// files that wrote standard attributes as UN (e.g. some RLE writers) read typed.
std::uint16_t un_vr_resolver(std::uint32_t tag) {
    const std::uint16_t group   = static_cast<std::uint16_t>(tag >> 16);
    const std::uint16_t element = static_cast<std::uint16_t>(tag & 0xFFFFu);
    // Private Creator (odd group, element 0x0010–0x00FF) is always LO (PS3.5
    // §7.8.1) — not in the public dict, so resolve it structurally.
    if ((group & 1u) && element >= 0x0010 && element <= 0x00FFu)
        return static_cast<std::uint16_t>('L' | ('O' << 8));   // libdcm VR_LO
    const auto info = dcmbase::dict::lookup(tag);
    if (!info.found || info.vr.size() != 2) return 0;
    return static_cast<std::uint16_t>(
        static_cast<std::uint8_t>(info.vr[0]) |
        (static_cast<std::uint8_t>(info.vr[1]) << 8));
}

// Creator-aware UN resolver for PRIVATE data elements, backed by dcmbasexx::dict's
// 12,608 vendor patterns. Files like
// AGFA's, which write private attributes as UN, read typed.
std::uint16_t un_vr_private_resolver(const char* creator, std::uint32_t creator_len,
                                     std::uint16_t group, std::uint8_t elem_low) {
    const auto info = dcmbase::dict::lookup_private(
        std::string_view(creator, creator_len), group, elem_low);
    if (!info.found || info.vr.size() != 2) return 0;
    return static_cast<std::uint16_t>(
        static_cast<std::uint8_t>(info.vr[0]) |
        (static_cast<std::uint8_t>(info.vr[1]) << 8));
}

// {tag, vr} of the dataset's bulk pixel-data element, or {0, ""} if none. Covers all three:
// (7FE0,0010) PixelData (OB/OW), (7FE0,0008) FloatPixelData (OF), (7FE0,0009) DoubleFloat (OD).
std::pair<std::uint32_t, std::string> compute_pixel_data_vr(std::span<const std::byte> bytes);

std::string read_json(const std::string& path, const std::string& charset_override,
                      bool inline_binary) {
    const auto bytes = slurp(path);
    auto j = dcmcore::bridge::to_dicom_json(std::span<const std::byte>{bytes},
                                            charset_override, inline_binary,
                                            &un_vr_resolver, &un_vr_private_resolver,
                                            /*numeric_text_as_string=*/true);
    if (!j) throw std::runtime_error("read_json " + path + ": "
                                     + std::string(dcmcore::bridge::describe(j.error())));
    std::string js = std::move(*j);
    // to_dicom_json drops bulk pixel data, so the pixel element's VR would be lost. Recover
    // it from the SAME buffer (no second file read, no pixel copy) and inject a value-less
    // {"vr":...} stub for whichever pixel tag is present — pydcm builds a lazy element from
    // it, keeping the file's real VR instead of guessing. (Front-inserted; build re-sorts.)
    const auto [ptag, pvr] = compute_pixel_data_vr(std::span<const std::byte>{bytes});
    if (ptag && !pvr.empty() && js.size() >= 2 && js.front() == '{') {
        char key[9];
        std::snprintf(key, sizeof key, "%08X", ptag);
        std::string stub = "\"" + std::string(key, 8) + "\":{\"vr\":\"" + pvr + "\"}";
        js.insert(std::size_t{1}, js[1] == '}' ? stub : stub + ",");
    }
    return js;
}

// read_meta_json(path) -> the FULL group-0002 File Meta Information as a DICOM JSON
// Model string (every (0002,xxxx), not just the 3 mandatory UIDs) — reusing the SAME
// bridge::dataset_to_json the dataset path uses, no separate parser. Empty for a naked
// file (no group-2). The meta group is always Explicit VR LE (PS3.10 §7.1), and sits
// at [132, dataset_offset) — right after the 128-byte preamble + "DICM".
std::string read_meta_json(const std::string& path) {
    const auto bytes = slurp(path);
    auto meta = dcmcore::part10::peek_file_meta(std::span<const std::byte>{bytes});
    if (!meta || meta->dataset_offset < 132 || meta->dataset_offset > bytes.size())
        return {};
    const std::span<const std::byte> g2{bytes.data() + 132, meta->dataset_offset - 132};
    dcmcore::bridge::to_json_options opts;
    opts.explicit_vr        = true;
    opts.inline_binary      = true;   // keep (0002,0001) FileMetaInformationVersion (OB) etc.
    opts.resolve_un         = &un_vr_resolver;
    opts.resolve_un_private = &un_vr_private_resolver;
    auto j = dcmcore::bridge::dataset_to_json(g2, opts);
    return j ? std::move(*j) : std::string{};
}

// read_pixel_data(path) -> raw bulk pixel value bytes — (7FE0,0010) PixelData,
// (7FE0,0008) FloatPixelData, or (7FE0,0009) DoubleFloatPixelData (None when the file
// has none, or uses a transfer syntax this fast path doesn't cover — deflate / EVR-BE,
// both rare). Reuses dataset::parse; the JSON model deliberately drops bulk pixel data,
// so this backs the lazy ds.PixelData (raw OB/OW or encapsulated bytes).
nb::object read_pixel_data(const std::string& path) {
    std::vector<std::byte> bytes;
    const char* pd = nullptr;               // found pixel value (points into bytes)
    std::size_t      pn = 0;
    auto find = [&] {                        // pure C++: file IO + dataset walk
        bytes = slurp(path);
        std::size_t off = 0;
        bool implicit = false;
        auto meta = dcmcore::part10::peek_file_meta(std::span<const std::byte>{bytes});
        if (meta) {
            off = meta->dataset_offset;
            implicit = (meta->transfer_syntax_uid == UID_LittleEndianImplicitTransferSyntax);
        } else {                                   // naked dataset — sniff like read_file_meta
            bool ex = false, le = true;
            if (!dcm_sniff_naked_dataset(reinterpret_cast<const char*>(bytes.data()),
                                         bytes.size(), &off, &ex, &le) || !le)
                return;                            // big-endian / undetectable → not covered
            implicit = !ex;
        }
        if (off > bytes.size()) return;
        const auto enc = implicit ? dcmcore::dataset::encoding::implicit_vr_le
                                  : dcmcore::dataset::encoding::explicit_vr_le;
        std::vector<dcmcore::dataset::element_view> els;
        if (!dcmcore::dataset::parse(
                std::span<const std::byte>{bytes.data() + off, bytes.size() - off}, enc, els))
            return;
        for (const auto& e : els)
            if (e.tag == 0x7FE00010u || e.tag == 0x7FE00008u || e.tag == 0x7FE00009u) {
                pd = e.value.data(); pn = e.value.size();
                return;
            }
    };
    { nb::gil_scoped_release rel; find(); }
    if (pd == nullptr) return nb::none();
    return nb::bytes(pd, pn);                // copies before `bytes` dies
}

// has_pixel_data(path) -> bool: does the dataset contain (7FE0,0010) PixelData, WITHOUT
// copying the bytes? Backs pydcm's lazy presence of PixelData in the
// Dataset mapping protocol (`'PixelData' in ds`, len(ds), iteration) without eager load.
bool has_pixel_data(const std::string& path) {
    const auto bytes = slurp(path);
    std::size_t off = 0;
    bool implicit = false;
    auto meta = dcmcore::part10::peek_file_meta(std::span<const std::byte>{bytes});
    if (meta) {
        off = meta->dataset_offset;
        implicit = (meta->transfer_syntax_uid == UID_LittleEndianImplicitTransferSyntax);
    } else {
        bool ex = false, le = true;
        if (!dcm_sniff_naked_dataset(reinterpret_cast<const char*>(bytes.data()),
                                     bytes.size(), &off, &ex, &le) || !le)
            return false;
        implicit = !ex;
    }
    if (off > bytes.size()) return false;
    const auto enc = implicit ? dcmcore::dataset::encoding::implicit_vr_le
                              : dcmcore::dataset::encoding::explicit_vr_le;
    std::vector<dcmcore::dataset::element_view> els;
    if (!dcmcore::dataset::parse(
            std::span<const std::byte>{bytes.data() + off, bytes.size() - off}, enc, els))
        return false;
    for (const auto& e : els)
        if (e.tag == 0x7FE00010u || e.tag == 0x7FE00008u || e.tag == 0x7FE00009u) return true;
    return false;
}

// {tag, vr} of the dataset's bulk pixel-data element from an ALREADY-IN-MEMORY Part-10/naked
// buffer — without reading the file again and without copying or decoding pixels (the parse
// only records element views). Covers all three pixel tags (7FE0,0008 FloatPixelData,
// 7FE0,0009 DoubleFloatPixelData, 7FE0,0010 PixelData); the VR is whatever
// dcmcore::dataset::parse resolved — i.e. the SAME value `dcmdump` prints (the single source
// of truth): the on-disk VR for Explicit VR, the C dictionary's VR for Implicit VR. The
// parser always types pixel tags concretely (OB/OW/OF/OD, never 0/UN), so there is nothing to
// re-derive. Returns {0, ""} when there is no pixel data or the transfer syntax isn't covered
// here (deflate / EVR-BE).
std::pair<std::uint32_t, std::string> compute_pixel_data_vr(std::span<const std::byte> bytes) {
    std::size_t off = 0;
    bool implicit = false;
    if (auto meta = dcmcore::part10::peek_file_meta(bytes)) {
        const auto& ts = meta->transfer_syntax_uid;
        if (ts == UID_DeflatedExplicitVRLittleEndianTransferSyntax || ts == UID_BigEndianExplicitTransferSyntax) return {0, {}};  // deflate / EVR-BE
        off = meta->dataset_offset;
        implicit = (ts == UID_LittleEndianImplicitTransferSyntax);
    } else {
        bool ex = false, le = true;
        if (!dcm_sniff_naked_dataset(reinterpret_cast<const char*>(bytes.data()),
                                     bytes.size(), &off, &ex, &le) || !le)
            return {0, {}};
        implicit = !ex;
    }
    if (off > bytes.size()) return {0, {}};
    std::vector<dcmcore::dataset::element_view> els;
    if (!dcmcore::dataset::parse(
            std::span<const std::byte>{bytes.data() + off, bytes.size() - off},
            implicit ? dcmcore::dataset::encoding::implicit_vr_le
                     : dcmcore::dataset::encoding::explicit_vr_le, els))
        return {0, {}};
    for (const auto& e : els)
        if (e.tag == 0x7FE00008u || e.tag == 0x7FE00009u || e.tag == 0x7FE00010u) {
            const char v[2] = { static_cast<char>(e.vr & 0xFF),
                                static_cast<char>((e.vr >> 8) & 0xFF) };
            return (v[0] && v[1]) ? std::pair{e.tag, std::string(v, 2)}
                                  : std::pair<std::uint32_t, std::string>{0, {}};
        }
    return {0, {}};
}

// pixel_data_vr(path) -> the bulk pixel element's VR string, or None when absent / unsupported
// TS. A thin slurp + compute, used only on the cold ds.PixelData fallback; the dcmread hot
// path gets the VR from read_json's injected stub instead (no extra read).
nb::object pixel_data_vr(const std::string& path) {
    const auto [tag, vr] = compute_pixel_data_vr(std::span<const std::byte>{slurp(path)});
    return vr.empty() ? nb::none() : nb::object(nb::str(vr.c_str(), vr.size()));
}

// mint_uid(seed, root) -> a DICOM UID, reusing dcmbase::uid::mint (THE canonical
// generator the de-identifier + img2dcm/pdf2dcm encapsulators share — FNV-1a 128-bit
// under the 2.25 UUID-derived arc). Deterministic for a given seed; pass random
// entropy for a fresh UID. Backs pydcm.uid.generate_uid.
std::string mint_uid(const std::string& seed, const std::string& root) {
    return dcmbase::uid::mint(seed, root.empty() ? std::string_view{"2.25"}
                                                 : std::string_view{root});
}

// transcode(part10_bytes, target_ts) -> Part-10 bytes re-encoded in target_ts. Reuses
// dcmcore::transcode::part10 (the WADO encoder) — backs pydcm Dataset.compress.
// Supports the encapsulated lossless targets (JPEG2000 lossless .90, JPEG-LS .80, HTJ2K
// .201, JPEG-XL) + the verbatim-copy path when target==source. Raises on an unsupported
// target (e.g. RLE / lossy) so the Python layer can surface a clear message.
nb::bytes transcode(nb::bytes part10, const std::string& target_ts) {
    const std::span<const std::byte> in{
        reinterpret_cast<const std::byte*>(part10.c_str()), part10.size()};
    auto r = dcmcore::transcode::part10(in, target_ts);
    if (!r) throw std::runtime_error("transcode to " + target_ts
                                     + ": unsupported target or decode failure (err "
                                     + std::to_string(static_cast<int>(r.error())) + ")");
    return nb::bytes(reinterpret_cast<const char*>(r->data()), r->size());
}

// write_part10(json, transfer_syntax="") -> Part-10 file bytes. Reuses
// bridge::from_dicom_json. transfer_syntax = Explicit VR LE (1.2.840.10008.1.2.1)
// emits EVR-LE straight from the JSON's per-element VRs (private / dict-resolved VRs
// preserved); empty / IVR-LE keeps the historical Implicit VR output. So a from-scratch
// dataset honours its file_meta TransferSyntaxUID instead of always landing as Implicit VR.
nb::bytes write_part10(const std::string& json, const std::string& transfer_syntax) {
    const bool ex = (transfer_syntax == "1.2.840.10008.1.2.1");
    auto b = [&] {
        nb::gil_scoped_release rel;
        return dcmcore::bridge::from_dicom_json(json, ex);
    }();
    if (!b) throw std::runtime_error("write_part10: "
                                     + std::string(dcmcore::bridge::describe(b.error())));
    return nb::bytes(reinterpret_cast<const char*>(b->data()), b->size());
}

// encode_ivr(json) -> bare Implicit VR LE DATASET bytes (no preamble/meta, no SOP-UID
// requirement). Reuses bridge::json_to_ivr — backs DIMSE identifier/query encoding
// (C-FIND/MOVE/GET keys), which is a dataset, not a Part-10 instance.
nb::bytes encode_ivr(const std::string& json) {
    auto b = dcmcore::bridge::json_to_ivr(json);
    if (!b) throw std::runtime_error("encode_ivr: "
                                     + std::string(dcmcore::bridge::describe(b.error())));
    return nb::bytes(reinterpret_cast<const char*>(b->data()), b->size());
}

// edit_part10(original_bytes, ops) -> edited Part-10 bytes. Production-grade
// save_as: applies (modify/insert/erase) ops to the ORIGINAL file byte-verbatim
// via dcmbase::edit, so the Transfer Syntax, PixelData, sequences and every
// untouched element survive exactly (the metadata-JSON write path can't — it
// drops pixels). ops = list of (tag:int, kind:str, value:str, vr:int).
nb::bytes edit_part10(nb::bytes original, nb::list ops) {
    std::vector<dcmbase::edit::edit_op> eops;
    eops.reserve(nb::len(ops));
    for (auto h : ops) {
        auto t = nb::cast<nb::tuple>(h);
        const std::string kind = nb::cast<std::string>(t[1]);
        const auto k = kind == "erase"  ? dcmbase::edit::op_kind::erase
                     : kind == "modify" ? dcmbase::edit::op_kind::modify
                                        : dcmbase::edit::op_kind::insert;
        // The value may be a Python str (text/numeric) or bytes (raw binary VR value,
        // e.g. PixelData) — std::string holds either byte-verbatim.
        std::string val;
        if (nb::isinstance<nb::bytes>(t[2])) {
            auto b = nb::cast<nb::bytes>(t[2]);
            val.assign(b.c_str(), b.size());
        } else {
            val = nb::cast<std::string>(t[2]);
        }
        eops.push_back({nb::cast<std::uint32_t>(t[0]), k, std::move(val),
                        nb::cast<std::uint16_t>(t[3])});
    }
    const std::span<const std::byte> file{
        reinterpret_cast<const std::byte*>(original.c_str()), original.size()};
    auto r = dcmbase::edit::apply_part10(file, eops);
    if (!r) throw std::runtime_error(std::string("edit_part10: ")
                                     + dcmbase::edit::describe(r.error()));
    return nb::bytes(reinterpret_cast<const char*>(r->data()), r->size());
}

// --- De-identification (PS3.15 Annex E) ------------------------------------
// Thin binding over dcmbase::deident::session — the SAME conformant native
// engine behind the CLI. The full PS3.15
// Table E.1-1 profile (617 attributes, recursive into sequences, consistent
// collision-free UID remap, (0012,006x) stamps) lives in dcmbasexx; this just
// builds the options struct from a dict and runs the session. No reimplementation.
namespace {

dcmbase::deident::options build_deident_options(nb::dict d) {
    dcmbase::deident::options o;
    auto sget = [&](const char* k, std::string& dst) {
        if (d.contains(k)) dst = nb::cast<std::string>(d[k]);
    };
    auto bget = [&](const char* k, bool& dst) {
        if (d.contains(k)) dst = nb::cast<bool>(d[k]);
    };
    sget("patient_name", o.patient_name);
    sget("patient_id",   o.patient_id);
    sget("uid_root",     o.uid_root);
    bget("retain_dates",          o.retain_full_dates);
    bget("retain_safe_private",   o.retain_safe_private);
    bget("clean_descriptors",     o.clean_descriptors);
    bget("retain_uids",           o.retain_uids);
    bget("retain_device_id",      o.retain_device_id);
    bget("retain_institution_id", o.retain_institution_id);
    bget("retain_patient_chars",  o.retain_patient_chars);
    bget("clean_graphics",        o.clean_graphics);
    bget("clean_struct_content",  o.clean_struct_content);
    bget("clean_pixel",           o.clean_pixel);  // stamp 113101 (pixanon does the work)
    if (d.contains("shift_dates_days"))
        o.shift_dates_days = nb::cast<int>(d["shift_dates_days"]);
    if (d.contains("profile")) {
        const auto p = nb::cast<std::string>(d["profile"]);
        o.prof = (p == "none") ? dcmbase::deident::profile::none
                               : dcmbase::deident::profile::basic_annex_e;
    }
    // replace: {tag_int: value_str} — per-tag D overrides (caller wins).
    if (d.contains("replace")) {
        for (auto [k, v] : nb::cast<nb::dict>(d["replace"]))
            o.replace.push_back({nb::cast<std::uint32_t>(k),
                                 nb::cast<std::string>(v)});
    }
    // remove: [tag_int] — tags to erase.
    if (d.contains("remove")) {
        for (auto t : nb::cast<nb::list>(d["remove"]))
            o.remove.push_back(nb::cast<std::uint32_t>(t));
    }
    return o;
}

// Run the burned-in pixel blackout (CTP signature library) over already
// de-identified bytes. A non-image instance (SR/KOS/…) decodes to no frames →
// pass it through unchanged. Other errors propagate.
nb::bytes pix_clean_bytes(const std::byte* p, std::size_t n, const char* who) {
    dcmbase::pixanon::options po;
    po.ruleset = dcmbase::pixanon::ctp_ruleset();
    auto r = dcmbase::pixanon::clean(std::span<const std::byte>{p, n}, po);
    if (r)
        return nb::bytes(reinterpret_cast<const char*>(r->first.data()), r->first.size());
    if (r.error() == dcmbase::pixanon::error::decode_failed)
        return nb::bytes(reinterpret_cast<const char*>(p), n);
    throw std::runtime_error(std::string(who) + ": clean_pixel: "
                             + dcmbase::pixanon::describe(r.error()));
}

nb::bytes deident_run(dcmbase::deident::session& sess, nb::bytes data,
                      bool clean_pixel, const char* who) {
    const std::span<const std::byte> file{
        reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    auto r = sess.process(file);
    if (!r) throw std::runtime_error(std::string(who) + ": "
                                     + dcmbase::deident::describe(r.error()));
    if (clean_pixel) return pix_clean_bytes(r->data(), r->size(), who);
    return nb::bytes(reinterpret_cast<const char*>(r->data()), r->size());
}

bool opt_clean_pixel(nb::dict d) {
    return d.contains("clean_pixel") && nb::cast<bool>(d["clean_pixel"]);
}

}  // namespace

// deidentify(data, options) -> de-identified Part-10 bytes. One file = one
// session. `options` is a dict of the deident::options fields (see deident.py).
nb::bytes deidentify(nb::bytes data, nb::dict options) {
    dcmbase::deident::session sess(build_deident_options(options));
    return deident_run(sess, data, opt_clean_pixel(options), "deidentify");
}

// deidentify_series(files, options) -> list of de-identified Part-10 bytes,
// processed through ONE session so the UID remap stays consistent across the
// batch (a study's instances keep their intra-study cross-references) — a single
// session per study.
nb::list deidentify_series(nb::list files, nb::dict options) {
    dcmbase::deident::session sess(build_deident_options(options));
    const bool cp = opt_clean_pixel(options);
    nb::list out;
    for (auto f : files)
        out.append(deident_run(sess, nb::cast<nb::bytes>(f), cp, "deidentify_series"));
    return out;
}

// clean_pixel_data(data, regions, use_ctp, require_match) -> Part-10 bytes with
// burned-in regions blacked out. Standalone pixel anonymizer (no tag scrub):
// `regions` is a list of (x, y, w, h[, frame]) pixel rects always applied;
// `use_ctp` also matches the CTP device-signature library. Re-emits uncompressed
// (never re-compresses). Raises on no-match only when require_match is set.
nb::bytes clean_pixel_data(nb::bytes data, nb::object regions,
                           bool use_ctp, bool require_match) {
    dcmbase::pixanon::options po;
    if (use_ctp) po.ruleset = dcmbase::pixanon::ctp_ruleset();
    po.require_match = require_match;
    if (!regions.is_none()) {
        for (auto r : nb::cast<nb::list>(regions)) {
            auto t = nb::cast<nb::tuple>(r);
            dcmbase::pixanon::region g;
            g.x = nb::cast<float>(t[0]); g.y = nb::cast<float>(t[1]);
            g.w = nb::cast<float>(t[2]); g.h = nb::cast<float>(t[3]);
            if (t.size() >= 5) g.frame = nb::cast<int>(t[4]);
            po.extra_regions.push_back(g);
        }
    }
    const std::span<const std::byte> file{
        reinterpret_cast<const std::byte*>(data.c_str()), data.size()};
    auto res = dcmbase::pixanon::clean(file, po);
    if (!res) throw std::runtime_error(std::string("clean_pixel_data: ")
                                       + dcmbase::pixanon::describe(res.error()));
    return nb::bytes(reinterpret_cast<const char*>(res->first.data()),
                     res->first.size());
}

// build_dicomdir(inputs, file_set_id) -> DICOMDIR Part-10 bytes. `inputs` is a
// list of (bytes, file_id) where file_id is the media-relative path (e.g.
// "PT0/ST0/SE0/IM0"). Thin binding over dcmbase::dicomdir::build — the engine
// reads each instance's key attributes, groups them into a PATIENT/STUDY/SERIES/
// leaf tree, and emits a conformant Explicit-VR-LE DICOMDIR with correct
// inter-record byte offsets.
nb::bytes build_dicomdir(nb::list inputs, const std::string& file_set_id) {
    // Hold the byte buffers + file-id strings alive across the build() call —
    // dicomdir::input borrows spans / string_views into them.
    std::vector<nb::bytes>   holders;
    std::vector<std::string> fids;
    holders.reserve(nb::len(inputs));
    fids.reserve(nb::len(inputs));
    for (auto h : inputs) {
        auto t = nb::cast<nb::tuple>(h);
        holders.push_back(nb::cast<nb::bytes>(t[0]));
        fids.push_back(nb::cast<std::string>(t[1]));
    }
    std::vector<dcmbase::dicomdir::input> ins;
    ins.reserve(holders.size());
    for (std::size_t i = 0; i < holders.size(); ++i) {
        ins.push_back({std::span<const std::byte>(
                           reinterpret_cast<const std::byte*>(holders[i].c_str()),
                           holders[i].size()),
                       fids[i]});
    }
    auto r = dcmbase::dicomdir::build(ins, {.file_set_id = file_set_id});
    if (!r) throw std::runtime_error(std::string("build_dicomdir: ")
                                     + std::string(dcmbase::dicomdir::describe(r.error())));
    return nb::bytes(reinterpret_cast<const char*>(r->bytes.data()), r->bytes.size());
}

// read_file_meta(path) -> {has_meta, transfer_syntax, sop_class, sop_instance}.
// The DICOM JSON Model omits group 0002, so the Dataset gets its
// ds.file_meta here. Reuses dcmcore::part10::peek_file_meta; for naked datasets
// (no file-meta) the Transfer Syntax is sniffed from the dataset bytes — the
// same detection dcm_decode runs — so ds.file_meta.TransferSyntaxUID still works.
nb::dict read_file_meta(const std::string& path) {
    const auto bytes = slurp(path);
    nb::dict d;
    // Sniff a dataset region's encoding → Transfer Syntax UID ("" if undetectable).
    const auto sniff_ts = [](std::span<const std::byte> region) -> std::string {
        std::size_t off = 0; bool ex = false, le = true;
        if (!dcm_sniff_naked_dataset(reinterpret_cast<const char*>(region.data()),
                                     region.size(), &off, &ex, &le))
            return {};
        return !le ? UID_BigEndianExplicitTransferSyntax        // Explicit VR Big Endian
             : ex  ? UID_LittleEndianExplicitTransferSyntax        // Explicit VR Little Endian
                   : UID_LittleEndianImplicitTransferSyntax;          // Implicit VR Little Endian
    };

    const std::span<const std::byte> all{bytes};
    auto meta = dcmcore::part10::peek_file_meta(all);
    if (meta) {                                    // real file-meta (strict or lenient peek)
        d["has_meta"]        = true;
        d["sop_class"]       = std::string(meta->sop_class_uid);
        d["sop_instance"]    = std::string(meta->sop_instance_uid);
        // "" when the meta omits (0002,0010) — no TS is reported there;
        // the dataset then parses as Implicit VR LE (handled in to_dicom_json).
        d["transfer_syntax"] = std::string(meta->transfer_syntax_uid);
        return d;
    }
    // No file-meta at all (truly naked / no DICM) — sniff the whole buffer.
    d["has_meta"]        = false;
    d["transfer_syntax"] = sniff_ts(all);
    return d;
}

// ---- DICOM data dictionary --------------------------------------------------
// Reused verbatim from dcmbasexx::dict — the native union of the public
// dictionaries (17,699 entries). pydcm ships NO table of its own; this is the
// keyword↔tag↔VR backbone behind attribute access (ds.PatientName).

nb::dict info_to_dict(const dcmbase::dict::info& i) {
    nb::dict d;
    d["keyword"] = std::string(i.keyword);
    d["name"]    = std::string(i.name);
    d["vr"]      = std::string(i.vr);
    d["vm"]      = std::string(i.vm);
    d["retired"] = i.retired;
    return d;
}

// keyword "PatientName" -> packed tag (group<<16)|element, or None. Public only.
std::optional<std::uint32_t> tag_for_keyword(const std::string& keyword) {
    return dcmbase::dict::keyword_to_tag(keyword);
}

// packed tag -> {keyword, name, vr, vm, retired}, or None if not a known std tag.
nb::object describe_tag(std::uint32_t tag) {
    const auto i = dcmbase::dict::lookup(tag);
    return i.found ? nb::object(info_to_dict(i)) : nb::none();
}

// private tag -> {…}, resolved by its Private Creator + the in-block low byte.
nb::object describe_private(const std::string& creator, std::uint16_t group,
                            std::uint8_t elem_low) {
    const auto i = dcmbase::dict::lookup_private(creator, group, elem_low);
    return i.found ? nb::object(info_to_dict(i)) : nb::none();
}

// ---- DICOM UID dictionary ---------------------------------------------------
// Reused verbatim from dcmbasexx::uid — the metadata layer over the value
// registry in dcmcore/dicom_uid.h (the union of dicom_uid.h + PS3.6,
// 1984 entries incl. PS3.16/PS3.20 Context Group + SR Template UIDs). pydcm.uid /
// pydcm.sop_class / pydcm.ctxgroups project from this — no Python-side copy.

// uid -> {name, type, keyword, info, retired, cid}, or None if unknown.
nb::object uid_lookup(const std::string& uid) {
    const auto i = dcmbase::uid::lookup(uid);
    if (!i.found) return nb::none();
    nb::dict d;
    d["name"]    = std::string(i.name);
    d["type"]    = std::string(i.type);
    d["keyword"] = std::string(i.keyword);
    d["info"]    = std::string(i.note);
    d["retired"] = i.retired;
    d["cid"]     = i.cid;
    return nb::object(d);
}

// public keyword "CTImageStorage" -> uid, or "" if unknown.
std::string keyword_to_uid(const std::string& keyword) {
    return std::string(dcmbase::uid::uid_for_keyword(keyword));
}

// The whole table as a list of (uid, name, type, keyword, info, retired, cid)
// tuples — pydcm.uid + pydcm.ctxgroups build their registries from this.
nb::list uid_table() {
    nb::list out;
    const std::size_t n = dcmbase::uid::count();
    for (std::size_t k = 0; k < n; ++k) {
        const auto e = dcmbase::uid::at(k);
        out.append(nb::make_tuple(std::string(e.uid), std::string(e.name),
                                  std::string(e.type), std::string(e.keyword),
                                  std::string(e.note), e.retired, e.cid));
    }
    return out;
}

// content_json(path) -> str | None : the semantic JSON of a structured object
// (Segmentation / RT Structure Set / RT Plan / RT Dose / Presentation State /
// Waveform). Shares dcmbase::content::to_json with the dcm2content CLI. None
// when not a structured object.
nb::object content_json(const std::string& path, bool contours, bool control_points) {
    const auto bytes = slurp(path);
    rdr::decoded dec{std::span<const std::byte>{bytes}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const std::string j = dcmbase::content::to_json(
        dec.get(), {.contours = contours, .control_points = control_points});
    if (j.empty()) return nb::none();
    return nb::str(j.c_str(), j.size());
}

// sr_to_html(path) -> str : clinical-readable HTML of an SR document via the
// native dcmbase::sr::to_html (dcm_sr_html C core) — the same renderer behind the
// CLI dsr2html, so the markup byte-matches and any SR renders (not only TID 1500).
// Backs pydcm.sr_to_html.
nb::object sr_to_html(const std::string& path) {
    const auto bytes = slurp(path);
    rdr::decoded dec{std::span<const std::byte>{bytes}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const std::string h = dcmbase::sr::to_html(dec.get());
    return nb::str(h.c_str(), h.size());
}

// iod_validate(path) -> [{severity, tag, module, message}, ...] : IOD / module
// Type-1/2 conformance for the SOP Class in (0008,0016) — the "dciodvfy core"
// behind `dcmclient dcmvalidate`. Self-contained dcmbase::iod::validate; here we
// only resolve the dataset span + encoding from the file meta and wrap the
// findings. Empty list = conformant at this level.
nb::list iod_validate(const std::string& path) {
    const auto bytes = slurp(path);
    if (bytes.empty()) throw std::runtime_error("cannot read: " + path);
    const std::span<const std::byte> file{bytes.data(), bytes.size()};

    rdr::decoded dec{file};
    if (!dec || !dec.get())
        throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dicom_info_t* info = dec.get();

    nb::list out;
    std::string sop = info->sop_class_uid ? info->sop_class_uid : "";
    while (!sop.empty() && (sop.back() == ' ' || sop.back() == '\0')) sop.pop_back();
    if (sop.empty()) return out;   // no SOP Class → no IOD to validate against

    // Dataset bytes + encoding — the SAME buffer/encoding authority dcm_decode
    // established (mirrors dcmbasexx's content re-walk): a Deflated dataset uses
    // the inflated EVR-LE buffer; otherwise the Part-10 file-meta transfer
    // syntax selects implicit/explicit LE; big-endian is not decodable by the
    // validator (its encoding enum is LE-only), so it is skipped.
    constexpr std::string_view kIVR = "1.2.840.10008.1.2";
    constexpr std::string_view kBE  = "1.2.840.10008.1.2.2";
    std::span<const std::byte> dataset;
    auto enc = dcmbase::iod::encoding::explicit_le;
    if (info->parse_ctx.deflate.bytes && info->parse_ctx.deflate.bytes_size > 0) {
        dataset = {reinterpret_cast<const std::byte*>(info->parse_ctx.deflate.bytes),
                   info->parse_ctx.deflate.bytes_size};
        enc = dcmbase::iod::encoding::explicit_le;
    } else if (const auto meta = dcmcore::part10::peek_file_meta(file);
               meta && meta->dataset_offset < bytes.size()) {
        if (meta->transfer_syntax_uid == kBE) return out;   // not validatable here
        dataset = file.subspan(meta->dataset_offset);
        enc = (meta->transfer_syntax_uid == kIVR) ? dcmbase::iod::encoding::implicit_le
                                                   : dcmbase::iod::encoding::explicit_le;
    } else {
        const std::string_view ts =
            info->transfer_syntax_uid ? std::string_view{info->transfer_syntax_uid} : kIVR;
        if (ts == kBE) return out;
        dataset = file;                                     // no file meta → raw dataset
        enc = (ts == kIVR) ? dcmbase::iod::encoding::implicit_le
                           : dcmbase::iod::encoding::explicit_le;
    }

    for (const auto& iss : dcmbase::iod::validate(sop, dataset, enc)) {
        nb::dict d;
        d["severity"] = std::string(iss.severity);
        d["tag"]      = iss.tag;
        d["module"]   = iss.module;
        d["message"]  = iss.message;
        out.append(d);
    }
    return out;
}

// read_rtdose(path) -> (ndarray[depth, rows, cols] float32 dose, meta)
//
// Thin shim over dcmbase::rt::read_dose — the scaling (pixel × DoseGridScaling,
// double math), geometry (dcm_volume-convention affine) and DVH decode all run
// in C++; here we only wrap the result as NumPy + a meta dict.
nb::object read_rtdose(const std::string& path) {
    const auto bytes = slurp(path);
    auto res = dcmbase::rt::read_dose(std::span<const std::byte>{bytes});
    if (!res) throw std::runtime_error("read_rtdose: " + res.error() + ": " + path);
    dcmbase::rt::dose_grid& g = *res;

    const std::size_t voxels = std::size_t(g.cols) * g.rows * g.depth;
    auto* out = new std::vector<std::byte>(voxels * sizeof(float));
    std::memcpy(out->data(), g.dose.data(), out->size());
    std::size_t shape[3] = {g.depth, g.rows, g.cols};   // [D, H, W]
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::byte>*>(p); });
    nb::ndarray<nb::numpy> arr(out->data(), 3, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});

    nb::dict meta;
    meta["depth"] = g.depth; meta["rows"] = g.rows; meta["columns"] = g.cols;
    meta["dtype"] = "float32";
    meta["dose_units"]          = g.dose_units;
    meta["dose_type"]           = g.dose_type;
    meta["dose_summation_type"] = g.dose_summation_type;
    meta["dose_grid_scaling"]   = g.dose_grid_scaling;
    meta["max_dose"]            = g.max_dose;
    meta["sop_instance_uid"]    = g.sop_instance_uid;
    if (!g.referenced_rt_plan_sop_instance_uid.empty())
        meta["referenced_rt_plan_sop_instance_uid"] = g.referenced_rt_plan_sop_instance_uid;
    nb::list sp; sp.append(g.spacing[2]); sp.append(g.spacing[1]); sp.append(g.spacing[0]);  // [z,y,x]
    meta["spacing"] = sp;
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(g.affine[i]);
    meta["affine"] = aff;                          // voxel→world, column-major 4×4
    nb::list offs; for (float o : g.grid_frame_offsets) offs.append(o);
    meta["grid_frame_offsets"] = offs;
    meta["uniform_offsets"]    = g.uniform_offsets;
    nb::list dvhs;
    for (const auto& c : g.dvhs) {
        nb::dict d;
        d["dvh_type"]   = c.type;
        d["dose_units"] = c.dose_units;
        d["dose_type"]  = c.dose_type;
        // unconditional: 0 is a legal ROI number (an absent sequence also reads 0)
        d["referenced_roi_number"] = c.referenced_roi_number;
        d["dose_scaling"] = c.dose_scaling;
        d["minimum_dose"] = c.minimum_dose;
        d["maximum_dose"] = c.maximum_dose;
        d["mean_dose"]    = c.mean_dose;
        nb::list bw; for (double v : c.bin_widths) bw.append(v);
        nb::list vol; for (double v : c.volumes) vol.append(v);
        d["bin_widths"] = bw;
        d["volumes"]    = vol;
        dvhs.append(d);
    }
    meta["dvhs"] = dvhs;
    return nb::make_tuple(arr, meta);
}

// write_rtdose(dose[D,R,C] float64, geometry, semantics, ids) -> bytes
//
// Thin shim over dcmbase's dcm_rtdose_export engine: quantisation (round to
// unsigned ints with a DS-text-consistent DoseGridScaling) and the Part-10
// emit all run in C. The Python side (pydcm.rt.write_rtdose) only derives
// geometry from an affine and copies IDs from a reference dataset.
nb::bytes write_rtdose_py(nb::ndarray<const double, nb::c_contig> dose,
                          const std::vector<double>& origin,
                          const std::vector<double>& orientation,
                          double ps_row, double ps_col,
                          const std::vector<double>& offsets,
                          const std::string& units, const std::string& dose_type,
                          const std::string& summation, const std::string& ref_plan_uid,
                          const std::string& patient_name, const std::string& patient_id,
                          const std::string& study_uid, const std::string& study_date,
                          const std::string& series_uid, const std::string& frame_of_ref_uid,
                          double scaling, int bits) {
    if (dose.ndim() != 3) throw std::runtime_error("dose must be [depth, rows, cols]");
    const std::uint32_t D = (std::uint32_t)dose.shape(0), R = (std::uint32_t)dose.shape(1),
                        C = (std::uint32_t)dose.shape(2);
    if (origin.size() != 3 || orientation.size() != 6)
        throw std::runtime_error("origin must be 3 values, orientation 6");
    if (offsets.size() != D)
        throw std::runtime_error("grid_frame_offsets length must equal depth");

    float org[3], ori[6];
    for (int i = 0; i < 3; ++i) org[i] = (float)origin[i];
    for (int i = 0; i < 6; ++i) ori[i] = (float)orientation[i];
    std::vector<float> offs(offsets.begin(), offsets.end());

    rtdose_export_begin(patient_name.c_str(), patient_id.c_str(),
                        study_uid.c_str(), study_date.c_str(),
                        series_uid.c_str(), frame_of_ref_uid.c_str());
    rtdose_export_set_dose(units.c_str(), dose_type.c_str(), summation.c_str(),
                           ref_plan_uid.empty() ? nullptr : ref_plan_uid.c_str());
    if (rtdose_export_set_grid(dose.data(), C, R, D, org, ori,
                               (float)ps_col, (float)ps_row, offs.data(),
                               (std::uint16_t)bits, scaling) != 0) {
        rtdose_export_free();
        throw std::runtime_error("write_rtdose: invalid grid (zero dims or negative dose values)");
    }
    std::uint32_t out_size = 0;
    const std::uint8_t* buf = rtdose_export_finalize(&out_size);
    if (!buf || out_size == 0) {
        rtdose_export_free();
        throw std::runtime_error("write_rtdose: export produced no output");
    }
    nb::bytes out(reinterpret_cast<const char*>(buf), out_size);
    rtdose_export_free();
    return out;
}

// ---- Encapsulated Documents (write_encapsulated / read_encapsulated) -------
// Thin shims over dcmbase::encap — the SAME engine behind dcmencap/dcmdecap
// and pdf2dcm/dcm2pdf. Assembly, per-type module sets, detection and
// MIME-aware extraction all live there.
nb::bytes encapsulate_py(nb::bytes payload, const std::string& type,
                         const std::string& title, const std::string& mime,
                         const std::string& units, nb::dict ids) {
    const auto t = dcmbase::encap::type_from_name(type);
    if (!t) throw std::runtime_error("write_encapsulated: unknown type '" + type +
                                     "' (pdf|cda|stl|obj|mtl)");
    dcmbase::encap::options opt;
    opt.type  = *t;
    opt.title = title;
    opt.mime  = mime;
    if (!units.empty()) opt.units_value = opt.units_meaning = units;
    dcmbase::encap::identity id;
    const auto gid = [&](const char* k) -> std::string {
        if (!ids.contains(k)) return {};
        try {
            return nb::cast<std::string>(ids[k]);
        } catch (...) {
            throw std::runtime_error(std::string("write_encapsulated: ids['") + k +
                                     "'] must be a str");
        }
    };
    id.patient_name = gid("patient_name"); id.patient_id = gid("patient_id");
    id.birth_date = gid("birth_date");     id.sex = gid("sex");
    id.study_uid = gid("study_uid");       id.study_date = gid("study_date");
    id.study_time = gid("study_time");     id.study_id = gid("study_id");
    id.accession = gid("accession");       id.referring = gid("referring");
    id.series_uid = gid("series_uid");
    id.frame_of_reference_uid = gid("frame_of_reference_uid");
    if (const auto cs = gid("charset"); !cs.empty()) id.charset = cs;

    auto r = dcmbase::encap::encapsulate(
        {reinterpret_cast<const std::byte*>(payload.c_str()), payload.size()}, opt, id);
    if (!r) throw std::runtime_error("write_encapsulated: " + r.error());
    return nb::bytes(reinterpret_cast<const char*>(r->part10.data()), r->part10.size());
}

nb::dict read_encapsulated_py(const std::string& path) {
    const auto bytes = slurp(path);
    auto x = dcmbase::encap::extract(std::span<const std::byte>{bytes});
    if (!x) throw std::runtime_error("read_encapsulated: " + x.error() + ": " + path);
    nb::dict d;
    d["payload"] = nb::bytes(reinterpret_cast<const char*>(x->payload.data()),
                             x->payload.size());
    d["mime"]  = x->mime;
    d["title"] = x->title;
    d["sop_class_uid"]    = x->sop_class_uid;
    d["sop_instance_uid"] = x->sop_instance_uid;
    d["type"] = x->type ? nb::cast(std::string(dcmbase::encap::name_for(*x->type)))
                        : nb::cast(nb::none());
    return d;
}

nb::object encap_detect_py(const std::string& filename, nb::bytes head) {
    const auto t = dcmbase::encap::detect(
        filename, {reinterpret_cast<const std::byte*>(head.c_str()), head.size()});
    if (!t) return nb::none();
    return nb::cast(std::string(dcmbase::encap::name_for(*t)));
}

// compute_dvh(rtstruct_path, rtdose_path, roi, ...) -> dict
//
// Thin shim over dcmbase::rt::compute_dvh (the DVH engine —
// rasterisation, dose-plane interpolation, histogram, volume and statistics
// all run in C++). Differential/cumulative counts come back as float64 NumPy.
nb::dict compute_dvh_py(const std::string& rtstruct_path, const std::string& rtdose_path,
                        int roi, int limit, bool calculate_full_volume, double thickness) {
    if (roi < 0 || roi > 65535)
        throw std::runtime_error("compute_dvh: roi must be in [0, 65535] (ROI Number is IS/uint16)");
    const auto sbytes = slurp(rtstruct_path);
    const auto dbytes = slurp(rtdose_path);
    dcmbase::rt::dvh_options opt;
    opt.limit_cgy = limit;
    opt.calculate_full_volume = calculate_full_volume;
    opt.thickness_mm = thickness;
    auto res = dcmbase::rt::compute_dvh(std::span<const std::byte>{sbytes},
                                        std::span<const std::byte>{dbytes},
                                        static_cast<std::uint16_t>(roi), opt);
    if (!res) throw std::runtime_error("compute_dvh: " + res.error());
    dcmbase::rt::computed_dvh& c = *res;

    auto as_array = [](const std::vector<double>& v) {
        auto* out = new std::vector<double>(v);
        std::size_t shape[1] = {out->size()};
        nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
        return nb::ndarray<nb::numpy>(out->data(), 1, shape, owner, nullptr,
            nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 64, 1});
    };
    nb::dict d;
    d["counts"]     = as_array(c.counts);      // differential, cm³ per 1-cGy bin
    d["cumulative"] = as_array(c.cumulative);  // suffix-sum, cm³
    d["bin_width_gy"] = 0.01;
    d["volume"]     = c.volume_cm3;
    d["min"]        = c.min_gy;
    d["max"]        = c.max_gy;
    d["mean"]       = c.mean_gy;
    d["name"]       = c.roi_name;
    d["notes"]      = c.notes;
    return d;
}

// Build the engine params, including the IBSI preprocessing knobs.
static dcmbase::radiomics::params make_params(int bins, float range_min, float range_max,
                                              float bin_width, float resample_spacing,
                                              bool normalize, float normalize_scale,
                                              bool averaged, bool resegment, bool resegment_sigma,
                                              float reseg_min, float reseg_max, bool resample_bspline,
                                              float voxel_array_shift,
                                              const std::vector<int>& distances = {}) {
    dcmbase::radiomics::params p{bins, range_min, range_max};
    p.bin_width = bin_width;  p.resample_spacing = resample_spacing;
    p.normalize = normalize;  p.normalize_scale = normalize_scale;
    p.averaged  = averaged;
    p.resegment = resegment;  p.resegment_sigma = resegment_sigma;
    p.reseg_min = reseg_min;  p.reseg_max = reseg_max;
    p.resample_bspline = resample_bspline;
    p.voxel_array_shift = voxel_array_shift;
    if (!distances.empty()) p.glcm_distances = distances;     // else engine default {1}
    return p;
}

// nb::list of ints (GLCM Chebyshev distances) -> std::vector<int>; empty -> {} (default {1}).
static std::vector<int> to_distances(const nb::list& dl) {
    std::vector<int> d;
    for (auto x : dl) d.push_back(nb::cast<int>(x));
    return d;
}

// radiomics_features(pixels float32[(slices,)h,w], mask uint8 same shape, spacing,
// bins, range, + preprocessing) -> {feature_name: value}. Shares dcmbase::radiomics::
// extract with the dcmradiomics CLI; the caller passes already-decoded arrays.
nb::dict radiomics_features(
    nb::ndarray<const float, nb::c_contig> pixels,
    nb::ndarray<const std::uint8_t, nb::c_contig> mask,
    float spacing_x, float spacing_y, float spacing_z,
    int bins, float range_min, float range_max,
    float bin_width, float resample_spacing, bool normalize, float normalize_scale,
    nb::list log_sigmas, bool wavelet, bool averaged,
    bool resegment, bool resegment_sigma, float reseg_min, float reseg_max, bool resample_bspline,
    float voxel_array_shift, nb::list filters, nb::list distances) {
    const std::size_t nd = pixels.ndim();
    if ((nd != 2 && nd != 3) || mask.ndim() != nd)
        throw std::runtime_error("pixels and mask must be matching 2D or 3D arrays");
    for (std::size_t i = 0; i < nd; ++i)
        if (mask.shape(i) != pixels.shape(i))
            throw std::runtime_error("pixels and mask shapes differ");
    const std::size_t nslices = nd == 3 ? pixels.shape(0) : 1;
    const std::size_t h = pixels.shape(nd - 2), w = pixels.shape(nd - 1);
    const std::size_t npx = h * w;
    const float*        px = pixels.data();
    const std::uint8_t* mk = mask.data();
    std::vector<dcmbase::radiomics::slice> slices;
    slices.reserve(nslices);
    for (std::size_t s = 0; s < nslices; ++s)
        slices.push_back({px + s * npx, mk + s * npx, static_cast<int>(w), static_cast<int>(h),
                          spacing_x, spacing_y, spacing_z});
    const auto pp = make_params(bins, range_min, range_max,
                                bin_width, resample_spacing, normalize, normalize_scale, averaged,
                            resegment, resegment_sigma, reseg_min, reseg_max, resample_bspline,
                            voxel_array_shift, to_distances(distances));
    dcmbase::radiomics::filter_config fc;
    for (auto s : log_sigmas) fc.log_sigmas.push_back(nb::cast<float>(s));
    fc.wavelet = wavelet;
    for (auto f : filters) {
        const std::string n = nb::cast<std::string>(f);
        if (n == "square") fc.square = true;
        else if (n == "squareroot") fc.squareroot = true;
        else if (n == "logarithm") fc.logarithm = true;
        else if (n == "exponential") fc.exponential = true;
        else if (n == "gradient") fc.gradient = true;
        else if (n == "lbp2d") fc.lbp2d = true;
        else if (n == "lbp3d") fc.lbp3d = true;
    }
    const auto feats = !fc.any()
        ? dcmbase::radiomics::extract(slices, pp)
        : dcmbase::radiomics::extract_filtered(slices, pp, fc);
    nb::dict out;
    for (const auto& f : feats) out[f.name.c_str()] = f.value;
    return out;
}

// radiomics_features_prepared(pixels, mask, spacing, bins, range, + preprocessing) ->
// (features dict, roi dict). The SAME standard IBSI set as radiomics_features for the
// original image (no filter passes), PLUS the preprocessed + discretised grid the
// extractor ran over — image / mask / levels as (nz,h,w) numpy arrays + nb / spacing /
// range — so the Python layer can compute CUSTOM features over the SAME grid the IBSI
// set used. dcmbase::radiomics::extract_prepared fills both in one preprocessing pass.
nb::tuple radiomics_features_prepared(
    nb::ndarray<const float, nb::c_contig> pixels,
    nb::ndarray<const std::uint8_t, nb::c_contig> mask,
    float spacing_x, float spacing_y, float spacing_z,
    int bins, float range_min, float range_max,
    float bin_width, float resample_spacing, bool normalize, float normalize_scale,
    bool averaged, bool resegment, bool resegment_sigma, float reseg_min, float reseg_max,
    bool resample_bspline, float voxel_array_shift, nb::list distances) {
    const std::size_t nd = pixels.ndim();
    if ((nd != 2 && nd != 3) || mask.ndim() != nd)
        throw std::runtime_error("pixels and mask must be matching 2D or 3D arrays");
    for (std::size_t i = 0; i < nd; ++i)
        if (mask.shape(i) != pixels.shape(i))
            throw std::runtime_error("pixels and mask shapes differ");
    const std::size_t nslices = nd == 3 ? pixels.shape(0) : 1;
    const std::size_t h = pixels.shape(nd - 2), w = pixels.shape(nd - 1);
    const std::size_t npx = h * w;
    const float*        px = pixels.data();
    const std::uint8_t* mk = mask.data();
    std::vector<dcmbase::radiomics::slice> slices;
    slices.reserve(nslices);
    for (std::size_t s = 0; s < nslices; ++s)
        slices.push_back({px + s * npx, mk + s * npx, static_cast<int>(w), static_cast<int>(h),
                          spacing_x, spacing_y, spacing_z});
    const auto pp = make_params(bins, range_min, range_max, bin_width, resample_spacing,
                                normalize, normalize_scale, averaged, resegment, resegment_sigma,
                                reseg_min, reseg_max, resample_bspline, voxel_array_shift,
                                to_distances(distances));
    dcmbase::radiomics::prepared_roi roi;
    const auto feats = dcmbase::radiomics::extract_prepared(slices, pp, roi);

    nb::dict fout;
    for (const auto& f : feats) fout[f.name.c_str()] = f.value;

    // Heap-own the grid vectors; capsules free them when Python drops the arrays.
    const std::size_t shape[3] = {static_cast<std::size_t>(roi.nz),
                                  static_cast<std::size_t>(roi.h),
                                  static_cast<std::size_t>(roi.w)};
    auto* img = new std::vector<float>(std::move(roi.intensities));
    auto* msk = new std::vector<std::uint8_t>(std::move(roi.mask));
    auto* lvl = new std::vector<std::int32_t>(roi.levels.begin(), roi.levels.end());
    nb::capsule img_o(img, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    nb::capsule msk_o(msk, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
    nb::capsule lvl_o(lvl, [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });

    nb::dict rout;
    rout["image"]  = nb::ndarray<nb::numpy>(img->data(), 3, shape, img_o, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});
    rout["mask"]   = nb::ndarray<nb::numpy>(msk->data(), 3, shape, msk_o, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
    rout["levels"] = nb::ndarray<nb::numpy>(lvl->data(), 3, shape, lvl_o, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Int, 32, 1});
    rout["nb"]      = roi.nb;
    rout["spacing"] = nb::make_tuple(roi.spacing_x, roi.spacing_y, roi.spacing_z);
    rout["range"]   = nb::make_tuple(roi.range_min, roi.range_max);
    return nb::make_tuple(fout, rout);
}

// radiomics_file(image_bytes, mask_bytes, roi_min, roi_max, bins, range) ->
// {feature_name: value}, straight from an image DICOM. Decode → spacing → ROI
// (co-framed mask DICOM when `mask` is non-empty, else the [roi_min,roi_max]
// threshold) → IBSI extract all run in dcmbase::radiomics::extract_file — the SAME
// orchestration as the dcmradiomics CLI, so Python re-implements none of it.
nb::dict radiomics_file(nb::bytes image, nb::bytes mask,
                        float roi_min, float roi_max,
                        int bins, float range_min, float range_max,
                        float bin_width, float resample_spacing,
                        bool normalize, float normalize_scale,
                        nb::list log_sigmas, bool wavelet, bool averaged,
    bool resegment, bool resegment_sigma, float reseg_min, float reseg_max, bool resample_bspline,
    float voxel_array_shift, nb::list filters, nb::list distances) {
    const std::span<const std::byte> img{
        reinterpret_cast<const std::byte*>(image.c_str()), image.size()};
    const std::span<const std::byte> msk{
        reinterpret_cast<const std::byte*>(mask.c_str()), mask.size()};
    const dcmbase::radiomics::roi r{msk, roi_min, roi_max};
    dcmbase::radiomics::filter_config fc;
    for (auto s : log_sigmas) fc.log_sigmas.push_back(nb::cast<float>(s));
    fc.wavelet = wavelet;
    for (auto f : filters) {
        const std::string n = nb::cast<std::string>(f);
        if (n == "square") fc.square = true;
        else if (n == "squareroot") fc.squareroot = true;
        else if (n == "logarithm") fc.logarithm = true;
        else if (n == "exponential") fc.exponential = true;
        else if (n == "gradient") fc.gradient = true;
        else if (n == "lbp2d") fc.lbp2d = true;
        else if (n == "lbp3d") fc.lbp3d = true;
    }
    const auto fr = dcmbase::radiomics::extract_file(
        img, r, make_params(bins, range_min, range_max,
                            bin_width, resample_spacing, normalize, normalize_scale, averaged,
                            resegment, resegment_sigma, reseg_min, reseg_max, resample_bspline,
                            voxel_array_shift, to_distances(distances)), fc);
    nb::dict out;
    for (const auto& f : fr.features) out[f.name.c_str()] = f.value;
    return out;
}

// ===========================================================================
// DCE-MRI pharmacokinetic modelling (pydcm.dce). Single-curve ops call the
// dcmbase C core (dcm_dce.h) directly; the voxel-wise parameter map shares
// dcmbase::dce::fit_slice with the CLI/server. Python passes already-decoded
// arrays — no DICOM parsing here.
// ===========================================================================
static dce_model_t dce_parse_model(const std::string& s) {
    if (s == "tofts")  return DCE_MODEL_TOFTS;
    if (s == "patlak") return DCE_MODEL_PATLAK;
    return DCE_MODEL_EXT_TOFTS;  // default / "ext_tofts"
}

static nb::ndarray<nb::numpy> dce_f64_1d(std::vector<double>&& v) {
    auto* out = new std::vector<double>(std::move(v));
    std::size_t shape[1] = {out->size()};
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::ndarray<nb::numpy>(out->data(), 1, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 64, 1});
}
static nb::ndarray<nb::numpy> dce_f32_2d(std::vector<float>&& v, std::size_t h, std::size_t w) {
    auto* out = new std::vector<float>(std::move(v));
    std::size_t shape[2] = {h, w};
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return nb::ndarray<nb::numpy>(out->data(), 2, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1});
}

// dce_parker_aif(times_min[T]) -> plasma AIF[T] (mM). hct default 0 = verbatim Parker.
nb::ndarray<nb::numpy> dce_parker_aif(nb::ndarray<const double, nb::c_contig> times, double hct) {
    const int n = (int)times.size();
    std::vector<double> cp(n);
    if (n > 0) dce_aif_parker(times.data(), n, hct, cp.data());
    return dce_f64_1d(std::move(cp));
}

static dce_aif_t dce_parse_aif(const std::string& s) {
    if (s == "georgiou")     return DCE_AIF_GEORGIOU;
    if (s == "fritz_hansen" || s == "fritz-hansen") return DCE_AIF_FRITZ_HANSEN;
    if (s == "weinmann")     return DCE_AIF_WEINMANN;
    if (s == "mcgrath")      return DCE_AIF_MCGRATH;
    return DCE_AIF_PARKER;
}

// dce_population_aif(model, times_min[T], hct) -> plasma AIF[T] (mM).
nb::ndarray<nb::numpy> dce_population_aif(const std::string& model,
                                         nb::ndarray<const double, nb::c_contig> times,
                                         double hct) {
    const int n = (int)times.size();
    std::vector<double> cp(n);
    if (n > 0) dce_aif(dce_parse_aif(model), times.data(), n, hct, cp.data());
    return dce_f64_1d(std::move(cp));
}

// dce_forward(times[T], cp[T], model, ktrans, ve, vp) -> synthetic Ct[T] (mM).
nb::ndarray<nb::numpy> dce_forward_py(nb::ndarray<const double, nb::c_contig> times,
                                      nb::ndarray<const double, nb::c_contig> cp,
                                      const std::string& model, double ktrans,
                                      double ve, double vp) {
    const int n = (int)times.size();
    if ((int)cp.size() != n) throw std::runtime_error("times and cp must have equal length");
    std::vector<double> ct(n);
    if (n > 0) dce_forward(times.data(), n, cp.data(), dce_parse_model(model), ktrans, ve, vp, ct.data());
    return dce_f64_1d(std::move(ct));
}

// dce_signal_to_conc(signal[T], n_baseline, t1_0_s, tr_s, fa_deg, r1) -> conc[T] (mM).
nb::ndarray<nb::numpy> dce_signal_to_conc_py(nb::ndarray<const double, nb::c_contig> sig,
                                             int n_baseline, double t1_0_s, double tr_s,
                                             double fa_deg, double r1) {
    const int n = (int)sig.size();
    std::vector<double> c(n);
    const int rc = (n > 0) ? dce_signal_to_conc(sig.data(), n, n_baseline, t1_0_s, tr_s,
                                                 fa_deg, r1, c.data()) : 1;
    if (rc != 0) throw std::runtime_error("signal_to_conc: invalid input (code "
                                          + std::to_string(rc) + ")");
    return dce_f64_1d(std::move(c));
}

// dce_fit_curve(times[T], ct[T], cp[T], model, fit_delay, delay_lo, delay_hi)
//   -> {ktrans, ve, vp, rmse, iters, ok, delay}.
nb::dict dce_fit_curve(nb::ndarray<const double, nb::c_contig> times,
                       nb::ndarray<const double, nb::c_contig> ct,
                       nb::ndarray<const double, nb::c_contig> cp,
                       const std::string& model,
                       bool fit_delay, double delay_lo, double delay_hi) {
    const int n = (int)times.size();
    if ((int)ct.size() != n || (int)cp.size() != n)
        throw std::runtime_error("times, ct, cp must have equal length");
    const dce_model_t cm = dce_parse_model(model);
    dce_fit_result_t r = fit_delay
        ? dce_fit_delay(times.data(), ct.data(), n, cp.data(), cm, delay_lo, delay_hi)
        : dce_fit(times.data(), ct.data(), n, cp.data(), cm);
    nb::dict d;
    d["ktrans"] = r.ktrans; d["ve"] = r.ve; d["vp"] = r.vp;
    d["rmse"] = r.rmse; d["iters"] = r.iters; d["ok"] = (bool)r.ok;
    d["delay"] = r.delay;
    return d;
}

// dce_fit_map(series[T,H,W] f32, times[T], model, input, hct, measured_cp?, mask?,
//   t1_0_s, tr_s, fa_deg, r1, n_baseline, enhance_thresh, t1_map?)
//   -> {ktrans[H,W], ve, vp, rmse, fitted}. Shares dcmbase::dce::fit_slice with CLI/server.
nb::dict dce_fit_map(nb::ndarray<const float, nb::c_contig> series,
                     nb::ndarray<const double, nb::c_contig> times,
                     const std::string& model, const std::string& input, double hct,
                     std::optional<nb::ndarray<const double, nb::c_contig>> measured_cp,
                     std::optional<nb::ndarray<const std::uint8_t, nb::c_contig>> mask,
                     double t1_0_s, double tr_s, double fa_deg, double r1, int n_baseline,
                     float enhance_thresh,
                     std::optional<nb::ndarray<const float, nb::c_contig>> t1_map,
                     bool fit_delay, double delay_lo, double delay_hi) {
    if (series.ndim() != 3) throw std::runtime_error("series must be a 3D [T,H,W] float32 array");
    const std::size_t T = series.shape(0), H = series.shape(1), W = series.shape(2);
    if ((std::size_t)times.size() != T) throw std::runtime_error("times length must equal T");
    const std::size_t npx = H * W;

    dcmbase::dce::series s;
    s.width = (int)W; s.height = (int)H;
    s.times_min.assign(times.data(), times.data() + T);
    s.frames.resize(T);
    const float* base = series.data();
    for (std::size_t t = 0; t < T; ++t) s.frames[t] = base + t * npx;

    dcmbase::dce::fit_params p;
    p.m = (model == "tofts") ? dcmbase::dce::model::tofts
        : (model == "patlak") ? dcmbase::dce::model::patlak
        : dcmbase::dce::model::ext_tofts;
    p.input = (input == "spgr" || input == "spgr_signal")
                  ? dcmbase::dce::input_kind::spgr_signal
                  : dcmbase::dce::input_kind::concentration;
    p.aif.population = !measured_cp.has_value();
    p.aif.hct = hct;
    if (measured_cp) {
        if ((std::size_t)measured_cp->size() != T)
            throw std::runtime_error("measured AIF length must equal T");
        p.aif.measured.assign(measured_cp->data(), measured_cp->data() + T);
    }
    p.spgr.t1_0_s = t1_0_s; p.spgr.tr_s = tr_s; p.spgr.fa_deg = fa_deg;
    p.spgr.r1 = r1; p.spgr.n_baseline = n_baseline;
    p.enhance_thresh = enhance_thresh;
    if (mask) {
        if ((std::size_t)mask->size() != npx)
            throw std::runtime_error("mask size must equal H*W");
        p.mask = mask->data();
    }
    if (t1_map) {
        if ((std::size_t)t1_map->size() != npx)
            throw std::runtime_error("t1_map size must equal H*W");
        p.spgr.t1_map = t1_map->data();
    }
    p.fit_delay = fit_delay; p.delay_lo = delay_lo; p.delay_hi = delay_hi;

    dcmbase::dce::maps m = dcmbase::dce::fit_slice(s, p);
    nb::dict d;
    d["ktrans"] = dce_f32_2d(std::move(m.ktrans), H, W);
    d["ve"]     = dce_f32_2d(std::move(m.ve), H, W);
    d["vp"]     = dce_f32_2d(std::move(m.vp), H, W);
    d["rmse"]   = dce_f32_2d(std::move(m.rmse), H, W);
    if (fit_delay) d["delay"] = dce_f32_2d(std::move(m.delay), H, W);
    d["fitted"] = m.fitted;
    return d;
}

// dce_t1_map_vfa(volumes[F,H,W] f32, fa_deg[F], tr_s, mask?) -> {t1[H,W], m0, fitted}.
// VFA/DESPOT1 baseline-T1 map via dcmbase::dce::t1_map_vfa (shared with CLI/server).
nb::dict dce_t1_map_vfa(nb::ndarray<const float, nb::c_contig> volumes,
                        nb::ndarray<const double, nb::c_contig> fa_deg, double tr_s,
                        std::optional<nb::ndarray<const std::uint8_t, nb::c_contig>> mask) {
    if (volumes.ndim() != 3) throw std::runtime_error("volumes must be a 3D [F,H,W] float32 array");
    const std::size_t F = volumes.shape(0), H = volumes.shape(1), W = volumes.shape(2);
    if ((std::size_t)fa_deg.size() != F) throw std::runtime_error("fa_deg length must equal F");
    const std::size_t npx = H * W;
    std::vector<const float*> frames(F);
    const float* base = volumes.data();
    for (std::size_t f = 0; f < F; ++f) frames[f] = base + f * npx;
    const std::uint8_t* m = nullptr;
    if (mask) {
        if ((std::size_t)mask->size() != npx) throw std::runtime_error("mask size must equal H*W");
        m = mask->data();
    }
    dcmbase::dce::t1_maps r = dcmbase::dce::t1_map_vfa(
        std::span<const float* const>(frames.data(), F), (int)W, (int)H,
        std::span<const double>(fa_deg.data(), F), tr_s, m);
    nb::dict d;
    d["t1"]     = dce_f32_2d(std::move(r.t1), H, W);
    d["m0"]     = dce_f32_2d(std::move(r.m0), H, W);
    d["fitted"] = r.fitted;
    return d;
}

// ---- shared SEG-authoring helpers (write_seg + write_seg_fractional) -------
// One reference slice's geometry + source SOP reference.
struct seg_ref_slice {
    std::array<float, 3> origin{}; std::array<float, 6> orient{};
    float sc = 1, sr = 1, st = 0; std::string sop, cls;
};
struct seg_ref_geom {
    std::vector<seg_ref_slice> rs;
    dcmbase::seg::meta         meta;
    std::uint32_t              W = 0, H = 0;
};

// Decode the source series → per-slice geometry + demographics, sorted by position
// along the slice normal. Structured objects / multi-frame / mismatched dims are skipped.
seg_ref_geom decode_ref_slices(const std::vector<std::string>& reference_paths) {
    seg_ref_geom g;
    bool demo = false;
    for (const auto& p : reference_paths) {
        rdr::decoded d{std::span<const std::byte>{slurp(p)}};
        const dicom_info_t* info = d.get();
        if (!info || d.frame_count() != 1) continue;
        if (info->seg.count || info->rts.count || info->ps.ps_type || info->wf.count) continue;
        if (!demo) {
            g.W = info->columns; g.H = info->rows;
            g.meta.patient_name = cstr(info->patient_name); g.meta.patient_id = cstr(info->patient_id);
            g.meta.study_uid = cstr(info->study_instance_uid); g.meta.study_date = cstr(info->study_date);
            g.meta.frame_of_reference_uid = cstr(info->frame_of_reference_uid);
            g.meta.source_series_uid = cstr(info->series_instance_uid); demo = true;
        } else if (info->columns != g.W || info->rows != g.H) continue;
        seg_ref_slice s;
        for (int i = 0; i < 3; ++i) s.origin[i] = info->origin[i];
        for (int i = 0; i < 6; ++i) s.orient[i] = info->orientation[i];
        s.sc = info->pixel_spacing[1]; s.sr = info->pixel_spacing[0]; s.st = info->slice_thickness;
        s.sop = cstr(info->sop_instance_uid); s.cls = cstr(info->sop_class_uid);
        g.rs.push_back(std::move(s));
    }
    if (g.rs.empty()) throw std::runtime_error("no decodable single-frame reference slices");
    const auto& o0 = g.rs.front().orient;
    const float nx = o0[1]*o0[5] - o0[2]*o0[4], ny = o0[2]*o0[3] - o0[0]*o0[5],
                nz = o0[0]*o0[4] - o0[1]*o0[3];
    std::sort(g.rs.begin(), g.rs.end(), [&](const seg_ref_slice& a, const seg_ref_slice& b) {
        return a.origin[0]*nx + a.origin[1]*ny + a.origin[2]*nz
             < b.origin[0]*nx + b.origin[1]*ny + b.origin[2]*nz; });
    return g;
}

struct seg_defs { std::vector<dcmbase::seg::segment> segs; std::vector<std::uint16_t> label_ids; };

// Marshal a Python list of segment dicts → coded dcmbase::seg::segment + labelIDs.
seg_defs marshal_segments(nb::list segments_py) {
    auto code_of = [](nb::handle seg, const char* key) -> dcmbase::seg::code {
        dcmbase::seg::code c;
        nb::dict d = nb::cast<nb::dict>(seg);
        if (d.contains(key)) {
            nb::tuple t = nb::cast<nb::tuple>(d[key]);
            if (t.size() >= 3) {
                c.value = nb::cast<std::string>(t[0]);
                c.scheme = nb::cast<std::string>(t[1]);
                c.meaning = nb::cast<std::string>(t[2]);
            }
        }
        return c;
    };
    seg_defs r;
    for (nb::handle h : segments_py) {
        nb::dict d = nb::cast<nb::dict>(h);
        dcmbase::seg::segment seg;
        if (d.contains("label")) seg.label = nb::cast<std::string>(d["label"]);
        if (d.contains("rgb")) {
            nb::tuple t = nb::cast<nb::tuple>(d["rgb"]);
            if (t.size() >= 3) { seg.r = nb::cast<std::uint8_t>(t[0]);
                seg.g = nb::cast<std::uint8_t>(t[1]); seg.b = nb::cast<std::uint8_t>(t[2]); }
        }
        seg.category = code_of(h, "category");
        seg.type     = code_of(h, "type");
        seg.anatomic = code_of(h, "anatomic");
        if (d.contains("algorithm_type")) seg.algorithm_type = nb::cast<std::string>(d["algorithm_type"]);
        if (d.contains("algorithm_name")) seg.algorithm_name = nb::cast<std::string>(d["algorithm_name"]);
        const auto lid = d.contains("labelID")
            ? nb::cast<std::uint16_t>(d["labelID"])
            : static_cast<std::uint16_t>(r.label_ids.size() + 1);
        r.segs.push_back(std::move(seg));
        r.label_ids.push_back(lid);
    }
    if (r.segs.empty()) throw std::runtime_error("segments list is empty");
    return r;
}

// The built SEG blob → Part-10 bytes, or write `output` and return None.
nb::object emit_seg(const std::vector<std::byte>& out, const std::string& output,
                    const char* empty_msg) {
    if (out.empty()) throw std::runtime_error(empty_msg);
    if (!output.empty()) {
        std::ofstream f(output, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + output);
        f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
        return nb::none();
    }
    return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

// write_seg(reference_paths, labelmap uint16[(slices,)h,w], segments, output="")
// -> bytes (or None when output is written). Coded BINARY Segmentation via
// dcmbase::seg::build (the mkseg writer). labelmap value k = the segment with labelID k.
nb::object write_seg(const std::vector<std::string>& reference_paths,
                     nb::ndarray<const std::uint16_t, nb::c_contig> labelmap,
                     nb::list segments_py, const std::string& output) {
    const seg_ref_geom g = decode_ref_slices(reference_paths);
    const std::size_t nslices = g.rs.size(), npx = std::size_t{g.W} * g.H;

    const std::size_t nd = labelmap.ndim();
    if ((nd != 2 && nd != 3) ||
        labelmap.shape(nd - 2) != g.H || labelmap.shape(nd - 1) != g.W ||
        (nd == 3 ? labelmap.shape(0) : 1) != nslices)
        throw std::runtime_error("labelmap shape does not match the reference series");
    const std::uint16_t* lm = labelmap.data();

    const seg_defs sd = marshal_segments(segments_py);
    auto seg_for = [&](std::uint16_t v) -> int {
        if (!v) return -1;
        for (std::size_t k = 0; k < sd.label_ids.size(); ++k)
            if (sd.label_ids[k] == v) return static_cast<int>(k);
        return -1;
    };
    std::vector<std::vector<std::uint16_t>> remapped(nslices);
    std::vector<dcmbase::seg::slice>        slices(nslices);
    for (std::size_t s = 0; s < nslices; ++s) {
        remapped[s].assign(npx, 0);
        const std::uint16_t* src = lm + s * npx;
        for (std::size_t i = 0; i < npx; ++i)
            if (int k = seg_for(src[i]); k >= 0) remapped[s][i] = static_cast<std::uint16_t>(k + 1);
        auto& sl = slices[s]; const auto& r = g.rs[s];
        sl.labelmap = remapped[s].data(); sl.w = g.W; sl.h = g.H;
        sl.origin = r.origin; sl.orientation = r.orient;
        sl.spacing_col = r.sc; sl.spacing_row = r.sr; sl.slice_thickness = r.st;
        sl.ref_sop_instance_uid = r.sop; sl.ref_sop_class_uid = r.cls;
    }
    return emit_seg(dcmbase::seg::build(g.meta, sd.segs, slices), output,
                    "labelmap had no voxels matching any segment labelID");
}

// write_seg_fractional(reference_paths, maps uint8[nseg,(slices,)h,w], segments,
// fractional_type, max_value, output="") -> bytes/None. FRACTIONAL Segmentation
// (8-bit probability/occupancy per segment per slice) via seg::build_fractional.
// `maps` is segment-major: maps[seg, slice] is that segment's value map on that slice.
nb::object write_seg_fractional(const std::vector<std::string>& reference_paths,
                                nb::ndarray<const std::uint8_t, nb::c_contig> maps,
                                nb::list segments_py, int fractional_type,
                                std::uint16_t max_value, const std::string& output) {
    const seg_ref_geom g = decode_ref_slices(reference_paths);
    const seg_defs     sd = marshal_segments(segments_py);
    const std::size_t  nseg = sd.segs.size(), nslices = g.rs.size(), npx = std::size_t{g.W} * g.H;

    const std::size_t nd = maps.ndim();
    if ((nd != 3 && nd != 4) || maps.shape(0) != nseg ||
        (nd == 4 ? maps.shape(1) : 1) != nslices ||
        maps.shape(nd - 2) != g.H || maps.shape(nd - 1) != g.W)
        throw std::runtime_error("maps shape must be [nseg, (slices,) H, W] matching segments × series");
    const std::uint8_t* md = maps.data();

    std::vector<dcmbase::seg::fractional_frame> frames;
    frames.reserve(nseg * nslices);
    for (std::size_t s = 0; s < nseg; ++s)
        for (std::size_t z = 0; z < nslices; ++z) {
            const auto& r = g.rs[z];
            dcmbase::seg::fractional_frame fr;
            fr.segment = static_cast<std::uint32_t>(s);
            fr.values  = md + (s * nslices + z) * npx;   // segment-major
            fr.w = g.W; fr.h = g.H; fr.origin = r.origin; fr.orientation = r.orient;
            fr.spacing_col = r.sc; fr.spacing_row = r.sr; fr.slice_thickness = r.st;
            fr.ref_sop_instance_uid = r.sop; fr.ref_sop_class_uid = r.cls;
            frames.push_back(std::move(fr));
        }
    const auto type = fractional_type == 1 ? dcmbase::seg::fractional::occupancy
                                           : dcmbase::seg::fractional::probability;
    return emit_seg(dcmbase::seg::build_fractional(g.meta, sd.segs, type,
                                                   max_value ? max_value : 255, frames),
                    output, "no non-empty fractional frames");
}

// ---- Parametric Map authoring (write_paramap) + reading (paramap_meta) ------
// Author a float Parametric Map (SOP 1.1.30) from a real-valued array over the
// source series' geometry, via dcmbase's dcm_paramap_export engine (the Parametric Map export
// capability). Reuses the SEG reference-geometry helper. rwvm:
// {slope,intercept,units_code,units_scheme,units_meaning,label,explanation,
// quantity_value,quantity_scheme,quantity_meaning} — all optional.
nb::object write_paramap(const std::vector<std::string>& reference_paths,
                         nb::ndarray<const float, nb::c_contig> values,
                         nb::dict rwvm, int store_bits, int store_signed,
                         const std::string& output) {
    const seg_ref_geom g = decode_ref_slices(reference_paths);
    const std::size_t nslices = g.rs.size(), npx = std::size_t{g.W} * g.H;

    const std::size_t nd = values.ndim();
    if ((nd != 2 && nd != 3) ||
        values.shape(nd - 2) != g.H || values.shape(nd - 1) != g.W ||
        (nd == 3 ? values.shape(0) : 1) != nslices)
        throw std::runtime_error("values shape does not match the reference series");
    const float* vd = values.data();

    auto ds = [&](const char* k) -> std::string {
        return rwvm.contains(k) ? nb::cast<std::string>(rwvm[k]) : std::string{};
    };
    const double slope     = rwvm.contains("slope")     ? nb::cast<double>(rwvm["slope"])     : 1.0;
    const double intercept = rwvm.contains("intercept") ? nb::cast<double>(rwvm["intercept"]) : 0.0;

    // A fresh Series UID for the map (SOP Instance is minted in finalize), seeded
    // from the source so it is reproducible per-content.
    const std::string series_uid =
        dcmbase::uid::mint(g.meta.study_uid + "|" + g.meta.source_series_uid + "|PARAMAP");

    paramap_export_begin(g.meta.patient_name.c_str(), g.meta.patient_id.c_str(),
                         g.meta.study_uid.c_str(), g.meta.study_date.c_str(),
                         series_uid.c_str(), g.meta.frame_of_reference_uid.c_str());
    paramap_export_set_rwvm(slope, intercept,
                            ds("units_code").c_str(), ds("units_scheme").c_str(),
                            ds("units_meaning").c_str(), ds("label").c_str(),
                            ds("explanation").c_str(), ds("quantity_value").c_str(),
                            ds("quantity_scheme").c_str(), ds("quantity_meaning").c_str());
    if (store_bits == 8 || store_bits == 16)
        paramap_export_set_pixel_type((std::uint16_t)store_bits, store_signed);

    for (std::size_t z = 0; z < nslices; ++z) {
        const auto& r = g.rs[z];
        paramap_export_add_frame(vd + z * npx, g.W, g.H, r.origin.data(), r.orient.data(),
                                 r.sc, r.sr, r.st, r.sop.c_str(), r.cls.c_str());
    }
    std::uint32_t out_size = 0;
    const std::uint8_t* buf = paramap_export_finalize(&out_size);
    std::vector<std::byte> bytes;
    if (buf && out_size)
        bytes.assign(reinterpret_cast<const std::byte*>(buf),
                     reinterpret_cast<const std::byte*>(buf) + out_size);
    paramap_export_free();
    return emit_seg(bytes, output, "parametric map export produced no output");
}

// paramap_meta(path) -> {is_parametric_map, pixel_data_vr, is_float, rwvm:{...}|None}.
// Surfaces the Real World Value Mapping (dcmcore parses it into info->rwv_mappings)
// + the float-pixel flag, so read_paramap can return real-world values.
nb::object paramap_meta(const std::string& path) {
    rdr::decoded dec{std::span<const std::byte>{slurp(path)}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dicom_info_t* info = dec.get();
    nb::dict d;
    d["is_parametric_map"] = (cstr(info->sop_class_uid) == std::string(UID_ParametricMapStorage));
    const int pvr = info->parse_ctx.pixel_data_vr;
    d["pixel_data_vr"] = pvr;
    d["is_float"]      = (pvr == 1 || pvr == 2);
    if (info->rwv_count > 0 && info->rwv_mappings) {
        const dcm_rwv_mapping_t& m = info->rwv_mappings[0];
        nb::dict r;
        r["slope"] = m.slope; r["intercept"] = m.intercept;
        r["first_value_mapped"] = m.first_mapped; r["last_value_mapped"] = m.last_mapped;
        r["label"] = cstr(m.label); r["units"] = cstr(m.units);
        r["has_lut"] = (m.lut_data != nullptr && m.lut_data_size > 0);
        d["rwvm"] = r;
    } else {
        d["rwvm"] = nb::none();
    }
    return d;
}

// ---- Segmentation reading (read_seg = segimage2itkimage) -------------------
// Segment terminology + geometry shared by the labelmap and masks return shapes.
void seg_add_meta(nb::dict& meta, const dicom_info_t* info, const seg_geometry_t& geom) {
    const seg_document_t& seg = info->seg;
    meta["segmentation_type"] = seg.segmentation_type == SEG_TYPE_FRACTIONAL ? "FRACTIONAL"
                              : seg.segmentation_type == SEG_TYPE_LABELMAP   ? "LABELMAP" : "BINARY";
    auto code3 = [](const char* v, const char* s, const char* m) {
        nb::dict c; c["value"] = cstr(v); c["scheme"] = cstr(s); c["meaning"] = cstr(m); return c;
    };
    nb::list segs;
    for (std::uint16_t k = 0; k < seg.count; ++k) {
        const seg_segment_t& s = seg.segments[k];
        nb::dict d;
        d["number"]   = s.number;
        d["label"]    = cstr(s.label);
        d["category"] = code3(s.category_value, s.category_scheme, s.category_meaning);
        d["type"]     = code3(s.property_value, s.property_scheme, s.property_meaning);
        d["anatomic"] = code3(s.anatomic_value, s.anatomic_scheme, s.anatomic_meaning);
        nb::list rgb; for (int i = 0; i < 3; ++i) rgb.append(s.display_rgb[i]);
        d["rgb"] = rgb;
        if (s.tracking_id)  d["tracking_id"]  = cstr(s.tracking_id);
        if (s.tracking_uid) d["tracking_uid"] = cstr(s.tracking_uid);
        segs.append(d);
    }
    meta["segments"] = segs;

    nb::list iop; for (int i = 0; i < 6; ++i) iop.append(geom.orientation[i]);
    meta["image_orientation_patient"] = iop;                       // [row dir, col dir]
    nb::list ps; ps.append(geom.spacing_row); ps.append(geom.spacing_col);
    meta["pixel_spacing"] = ps;                                    // [row, col] mm
    meta["slice_thickness"] = geom.slice_thickness;
    nb::list origins;
    for (std::uint32_t z = 0; z < geom.slices; ++z) {
        nb::list o; for (int i = 0; i < 3; ++i) o.append(geom.slice_origins[z*3 + i]);
        origins.append(o);
    }
    meta["slice_origins"] = origins;                              // ImagePositionPatient per slice
    nb::list aff; for (int i = 0; i < 16; ++i) aff.append(geom.model_matrix[i]);
    meta["affine"] = aff;                                         // voxel→world LPS, column-major 4×4
                                                                  //   (the engine's single canonical affine)
    meta["slices"] = geom.slices; meta["rows"] = geom.rows; meta["columns"] = geom.cols;
    meta["series_instance_uid"]    = cstr(info->series_instance_uid);
    meta["study_instance_uid"]     = cstr(info->study_instance_uid);
    meta["frame_of_reference_uid"] = cstr(info->frame_of_reference_uid);
}

// read_seg(path, masks=False) -> (array, meta) | None. Reconstructs a DICOM
// Segmentation to a labelmap (segment numbers, [slices,rows,cols] uint16) or, with
// masks=True, per-segment occupancy ([nseg,slices,rows,cols] float32 in [0,1]) — the
// SEG decode capability, over dcmbase's dcm_seg_decode engine.
nb::object read_seg(const std::string& path, bool masks) {
    const auto bytes = slurp(path);
    rdr::decoded dec{std::span<const std::byte>{bytes}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dicom_info_t* info = dec.get();
    if (info->seg.count == 0) return nb::none();   // not a segmentation

    nb::dict meta;
    if (masks) {
        seg_masks_t m;
        if (seg_decode_masks(info, &m) != 0) throw std::runtime_error("SEG mask decode failed: " + path);
        const std::size_t shape[4] = {m.num_segments, m.geom.slices, m.geom.rows, m.geom.cols};
        const std::size_t n = shape[0] * shape[1] * shape[2] * shape[3];
        auto* buf = new std::vector<float>(m.masks, m.masks + n);
        seg_add_meta(meta, info, m.geom);
        nb::list snums; for (std::uint32_t k = 0; k < m.num_segments; ++k) snums.append(m.segment_numbers[k]);
        meta["segment_numbers"] = snums;                          // plane k ↔ this Segment Number
        seg_masks_free(&m);
        nb::capsule owner(buf, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
        nb::dlpack::dtype dt{(std::uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
        nb::ndarray<nb::numpy> arr(buf->data(), 4, shape, owner, nullptr, dt);
        return nb::make_tuple(arr, meta);
    }
    seg_labelmap_t lm;
    if (seg_decode_labelmap(info, &lm) != 0) throw std::runtime_error("SEG decode failed: " + path);
    const std::size_t shape[3] = {lm.geom.slices, lm.geom.rows, lm.geom.cols};
    const std::size_t n = shape[0] * shape[1] * shape[2];
    auto* buf = new std::vector<std::uint16_t>(lm.labelmap, lm.labelmap + n);
    seg_add_meta(meta, info, lm.geom);
    meta["overlapping"] = (bool)lm.overlapping;
    seg_labelmap_free(&lm);
    nb::capsule owner(buf, [](void* p) noexcept { delete static_cast<std::vector<std::uint16_t>*>(p); });
    nb::dlpack::dtype dt{(std::uint8_t)nb::dlpack::dtype_code::UInt, 16, 1};
    nb::ndarray<nb::numpy> arr(buf->data(), 3, shape, owner, nullptr, dt);
    return nb::make_tuple(arr, meta);
}

// ---- SR authoring (write_sr) ----------------------------------------------
std::string sr_str(nb::dict d, const char* k) {
    return d.contains(k) ? nb::cast<std::string>(d[k]) : std::string{};
}
dcmbase::sr::code sr_code(nb::handle h) {
    dcmbase::sr::code c;
    if (h && nb::isinstance<nb::dict>(h)) {
        nb::dict d = nb::cast<nb::dict>(h);
        c.value = sr_str(d, "value"); c.scheme = sr_str(d, "scheme"); c.meaning = sr_str(d, "meaning");
    }
    return c;
}
int sr_vt(const std::string& s) {
    if (s == "TEXT") return SR_VT_TEXT;       if (s == "CODE") return SR_VT_CODE;
    if (s == "NUM") return SR_VT_NUM;         if (s == "DATE") return SR_VT_DATE;
    if (s == "TIME") return SR_VT_TIME;       if (s == "DATETIME") return SR_VT_DATETIME;
    if (s == "UIDREF") return SR_VT_UIDREF;   if (s == "PNAME") return SR_VT_PNAME;
    if (s == "CONTAINER") return SR_VT_CONTAINER; if (s == "IMAGE") return SR_VT_IMAGE;
    if (s == "COMPOSITE") return SR_VT_COMPOSITE; if (s == "WAVEFORM") return SR_VT_WAVEFORM;
    if (s == "SCOORD") return SR_VT_SCOORD;   if (s == "SCOORD3D") return SR_VT_SCOORD3D;
    if (s == "TCOORD") return SR_VT_TCOORD;   return SR_VT_TEXT;
}
int sr_rel(const std::string& s) {
    if (s == "HAS OBS CONTEXT") return SR_REL_HAS_OBS_CONTEXT;
    if (s == "HAS ACQ CONTEXT") return SR_REL_HAS_ACQ_CONTEXT;
    if (s == "HAS CONCEPT MOD") return SR_REL_HAS_CONCEPT_MOD;
    if (s == "HAS PROPERTIES") return SR_REL_HAS_PROPERTIES;
    if (s == "INFERRED FROM") return SR_REL_INFERRED_FROM;
    if (s == "SELECTED FROM") return SR_REL_SELECTED_FROM;
    return SR_REL_CONTAINS;
}
int sr_gt(const std::string& s) {
    if (s == "POINT") return SR_GRAPHIC_POINT;  if (s == "MULTIPOINT") return SR_GRAPHIC_MULTIPOINT;
    if (s == "CIRCLE") return SR_GRAPHIC_CIRCLE; if (s == "ELLIPSE") return SR_GRAPHIC_ELLIPSE;
    return SR_GRAPHIC_POLYLINE;
}
void sr_flatten(nb::dict node, int parent, std::vector<dcmbase::sr::content_item>& items) {
    dcmbase::sr::content_item ci;
    ci.parent = parent;
    ci.relationship = sr_rel(sr_str(node, "relationship"));
    ci.value_type   = sr_vt(sr_str(node, "value_type"));
    if (node.contains("concept")) ci.concept_name = sr_code(node["concept"]);
    ci.text = sr_str(node, "text");
    if (node.contains("code")) ci.value_code = sr_code(node["code"]);
    if (node.contains("value")) ci.num = nb::cast<double>(node["value"]);
    if (node.contains("unit")) ci.unit = sr_code(node["unit"]);
    ci.datetime = sr_str(node, "datetime");                              // DATE/TIME/DATETIME value
    if (ci.datetime.empty()) ci.datetime = sr_str(node, "uid");          // UIDREF (content.to_json key)
    if (ci.datetime.empty()) ci.datetime = sr_str(node, "person_name");  // PNAME  (content.to_json key)
    ci.ref_sop_class = sr_str(node, "ref_sop_class");
    ci.ref_sop_instance = sr_str(node, "ref_sop_instance");
    ci.graphic_type = sr_gt(sr_str(node, "graphic_type"));
    if (sr_str(node, "continuity") == "CONTINUOUS") ci.continuity = 1;
    if (node.contains("graphic_data"))
        for (nb::handle v : nb::cast<nb::list>(node["graphic_data"]))
            ci.graphic_data.push_back(nb::cast<float>(v));
    ci.frame_of_reference_uid = sr_str(node, "frame_of_reference_uid");
    ci.observation_datetime = sr_str(node, "observation_datetime");
    const int my = static_cast<int>(items.size());
    items.push_back(std::move(ci));
    if (node.contains("content"))
        for (nb::handle ch : nb::cast<nb::list>(node["content"]))
            sr_flatten(nb::cast<nb::dict>(ch), my, items);
}

// write_sr(document, output="") -> bytes (or None). Authors a Comprehensive SR
// from a content-tree dict (same shape as the `mksr` JSON) via dcmbase::sr::build.
nb::object write_sr(nb::dict document, const std::string& output) {
    dcmbase::sr::document doc;
    doc.patient_name = sr_str(document, "patient_name");
    doc.patient_id = sr_str(document, "patient_id");
    doc.study_uid = sr_str(document, "study_uid");
    doc.study_date = sr_str(document, "study_date");
    doc.series_uid = sr_str(document, "series_uid");
    doc.sop_class_uid = sr_str(document, "sop_class_uid");
    doc.completion_flag = sr_str(document, "completion_flag");
    doc.verification_flag = sr_str(document, "verification_flag");

    std::vector<dcmbase::sr::content_item> items;
    dcmbase::sr::content_item root;
    root.parent = -1; root.value_type = SR_VT_CONTAINER;
    if (document.contains("title")) root.concept_name = sr_code(document["title"]);
    if (sr_str(document, "continuity") == "CONTINUOUS") root.continuity = 1;
    items.push_back(std::move(root));
    if (document.contains("content"))
        for (nb::handle ch : nb::cast<nb::list>(document["content"]))
            sr_flatten(nb::cast<nb::dict>(ch), 0, items);

    return emit_seg(dcmbase::sr::build(doc, items), output, "failed to build the SR document");
}

// ---- SR coded-concept lookup / validation (PS3.16) ---------
nb::object sr_code_meaning(const std::string& scheme, const std::string& value) {
    const std::string m = dcmbase::sr::code_meaning(scheme, value);
    return m.empty() ? nb::none() : nb::object(nb::str(m.c_str()));
}
bool sr_validate_code(const std::string& scheme, const std::string& value,
                      const std::string& meaning) {
    return dcmbase::sr::validate_code(scheme, value, meaning);
}
bool sr_cid_has(int cid, const std::string& scheme, const std::string& value) {
    return dcmbase::sr::cid_has(cid, scheme, value);
}

// sr_validate(path) -> [{severity, location, message}, ...] : structural + coded
// conformance of an SR's content tree (dcmbase::sr::validate).
nb::list sr_validate(const std::string& path) {
    rdr::decoded dec{std::span<const std::byte>{slurp(path)}};
    nb::list out;
    if (!dec) return out;
    for (const dcmbase::sr::issue& iss : dcmbase::sr::validate(dec.get())) {
        nb::dict d;
        d["severity"] = iss.severity;
        d["location"] = iss.location;
        d["message"] = iss.message;
        out.append(d);
    }
    return out;
}

// ---- TID 1500 Measurement Report (write_report / read_report) ---------------
// dcm_sr_export's OWN graphic-type numbering (0=POINT 1=POLYLINE 2=CIRCLE 3=ELLIPSE)
// — note this differs from SR_GRAPHIC_*; the pydcm API uses string names on both
// sides so the writer input and reader output round-trip unambiguously.
std::uint8_t report_gt(const std::string& s) {
    if (s == "POINT")   return 0;
    if (s == "CIRCLE")  return 2;
    if (s == "ELLIPSE") return 3;
    return 1;  // POLYLINE (covers RULER, the 2-point line)
}
const char* sr_graphic_name(std::uint8_t gt) {   // SR_GRAPHIC_* (parser side) → string
    switch (gt) {
    case SR_GRAPHIC_POINT:      return "POINT";
    case SR_GRAPHIC_MULTIPOINT: return "MULTIPOINT";
    case SR_GRAPHIC_CIRCLE:     return "CIRCLE";
    case SR_GRAPHIC_ELLIPSE:    return "ELLIPSE";
    default:                    return "POLYLINE";
    }
}

// write_report(document, output="") -> bytes (or None). Authors a TID 1500 Measurement
// Report SR from a measurements list via dcmbase's dcm_sr_export engine (the `mkreport`
// writer). document: {patient_name,patient_id,study_uid,study_date,series_uid,
// measurements:[{concept_value,concept_scheme,concept_meaning,value,unit_code,
// unit_meaning,ref_sop_class_uid,ref_sop_instance_uid,graphic_type(str),scoord:[...]}]}.
nb::object write_report(nb::dict document, const std::string& output) {
    if (!document.contains("measurements") || !nb::isinstance<nb::list>(document["measurements"]))
        throw std::runtime_error("document['measurements'] must be a list");
    nb::list meas = nb::cast<nb::list>(document["measurements"]);
    if (meas.size() == 0) throw std::runtime_error("document['measurements'] is empty");

    // Content-derived UIDs (reproducible, distinct per report) when not supplied —
    // mirrors the `mkreport` CLI's content-seeded minting.
    std::string seed;
    for (nb::handle h : meas) {
        nb::dict m = nb::cast<nb::dict>(h);
        seed += sr_str(m, "concept_value"); seed += '|';
        if (m.contains("value")) seed += std::to_string(nb::cast<double>(m["value"]));
        seed += ';';
    }
    std::string study_uid  = sr_str(document, "study_uid");
    std::string series_uid = sr_str(document, "series_uid");
    if (study_uid.empty())  study_uid  = dcmbase::uid::mint(seed + "|STUDY");
    if (series_uid.empty()) series_uid = dcmbase::uid::mint(seed + "|SERIES");

    sr_export_begin(sr_str(document, "patient_name").c_str(),
                    sr_str(document, "patient_id").c_str(),
                    study_uid.c_str(), sr_str(document, "study_date").c_str(),
                    series_uid.c_str());

    for (nb::handle h : meas) {
        nb::dict m = nb::cast<nb::dict>(h);
        std::vector<float> scoord;
        if (m.contains("scoord"))
            for (nb::handle v : nb::cast<nb::list>(m["scoord"])) scoord.push_back(nb::cast<float>(v));
        if (scoord.size() % 2) scoord.pop_back();   // SCOORD is (col,row) pairs; drop unpaired tail
        sr_export_add_measurement(
            sr_str(m, "concept_value").c_str(), sr_str(m, "concept_scheme").c_str(),
            sr_str(m, "concept_meaning").c_str(),
            m.contains("value") ? nb::cast<double>(m["value"]) : 0.0,
            sr_str(m, "unit_code").c_str(), sr_str(m, "unit_meaning").c_str(),
            sr_str(m, "ref_sop_class_uid").c_str(), sr_str(m, "ref_sop_instance_uid").c_str(),
            report_gt(sr_str(m, "graphic_type")),
            scoord.empty() ? nullptr : scoord.data(),
            static_cast<std::uint32_t>(scoord.size() / 2));
    }

    std::uint32_t out_size = 0;
    const std::uint8_t* buf = sr_export_finalize(&out_size);
    std::vector<std::byte> bytes;
    if (buf && out_size)
        bytes.assign(reinterpret_cast<const std::byte*>(buf),
                     reinterpret_cast<const std::byte*>(buf) + out_size);
    sr_export_free();
    return emit_seg(bytes, output, "SR export produced no output");
}

// read_report(path) -> {patient_name, patient_id, study_uid, study_date, series_uid,
// measurements:[...]} : the measurements of a TID 1500 Measurement Report SR, round-
// tripping write_report's input. Walks the parsed SR content tree (info->sr) for NUM
// nodes, pairing each with its SCOORD geometry + IMAGE reference (as dcm_sr_import
// navigates). Empty measurements list when `path` carries no SR content.
nb::object read_report(const std::string& path) {
    const auto bytes = slurp(path);
    rdr::decoded dec{std::span<const std::byte>{bytes}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dicom_info_t* info = dec.get();
    const sr_document_t& sr = info->sr;

    nb::dict doc;
    doc["patient_name"] = cstr(info->patient_name);
    doc["patient_id"]   = cstr(info->patient_id);
    doc["study_uid"]    = cstr(info->study_instance_uid);
    doc["study_date"]   = cstr(info->study_date);
    doc["series_uid"]   = cstr(info->series_instance_uid);

    nb::list measurements;
    for (std::uint16_t i = 0; i < sr.count; ++i) {
        const sr_node_t& n = sr.nodes[i];
        if (n.value_type != SR_VT_NUM) continue;

        nb::dict m;
        m["concept_value"]   = cstr(n.concept_value);
        m["concept_scheme"]  = cstr(n.concept_scheme);
        m["concept_meaning"] = cstr(n.concept_meaning);
        m["value"]           = n.num.value;
        m["unit_code"]       = cstr(n.num.unit_code);
        m["unit_meaning"]    = cstr(n.num.unit_meaning);

        // SCOORD child holds geometry; the IMAGE reference hangs under the SCOORD
        // (export nesting) or directly under the NUM — same lookup as dcm_sr_import.
        const sr_node_t* scoord = nullptr; std::int16_t scoord_idx = -1;
        for (std::uint16_t j = 0; j < sr.count; ++j)
            if (sr.nodes[j].parent_index == static_cast<std::int16_t>(i) &&
                sr.nodes[j].value_type == SR_VT_SCOORD) { scoord = &sr.nodes[j]; scoord_idx = static_cast<std::int16_t>(j); break; }

        const std::int16_t img_parent = scoord ? scoord_idx : static_cast<std::int16_t>(i);
        for (std::uint16_t j = 0; j < sr.count; ++j)
            if (sr.nodes[j].parent_index == img_parent && sr.nodes[j].value_type == SR_VT_IMAGE) {
                m["ref_sop_class_uid"]    = cstr(sr.nodes[j].ref.sop_class);
                m["ref_sop_instance_uid"] = cstr(sr.nodes[j].ref.sop_instance);
                break;
            }
        if (scoord) {
            m["graphic_type"] = sr_graphic_name(scoord->scoord.graphic_type);
            nb::list pts;
            const std::uint32_t fc = static_cast<std::uint32_t>(scoord->scoord.num_points) * 2;
            for (std::uint32_t k = 0; scoord->scoord.data && k < fc; ++k)
                pts.append(scoord->scoord.data[k]);
            m["scoord"] = pts;
        }
        measurements.append(m);
    }
    doc["measurements"] = measurements;
    return doc;
}

// ---- TID 1500 Measurement Report — typed template (write/read_measurement_report)
nb::dict code_dict(const dcmbase::sr::code& c) {
    nb::dict d; d["value"] = c.value; d["scheme"] = c.scheme; d["meaning"] = c.meaning; return d;
}

// finding_site dict: the site code fields directly + optional laterality /
// topographical_modifier sub-codes.
dcmbase::sr::finding_site fs_from(nb::handle h) {
    nb::dict d = nb::cast<nb::dict>(h);
    dcmbase::sr::finding_site fs;
    fs.site = sr_code(h);
    if (d.contains("laterality")) fs.laterality = sr_code(d["laterality"]);
    if (d.contains("topographical_modifier")) fs.topographical_modifier = sr_code(d["topographical_modifier"]);
    return fs;
}
nb::dict fs_dict(const dcmbase::sr::finding_site& fs) {
    nb::dict d = code_dict(fs.site);
    if (!fs.laterality.value.empty()) d["laterality"] = code_dict(fs.laterality);
    if (!fs.topographical_modifier.value.empty()) d["topographical_modifier"] = code_dict(fs.topographical_modifier);
    return d;
}
dcmbase::sr::sr_roi roi_from(nb::dict roi) {
    dcmbase::sr::sr_roi r;
    r.graphic_type = sr_gt(sr_str(roi, "graphic_type"));
    r.is_3d = roi.contains("is_3d") && nb::cast<bool>(roi["is_3d"]);
    if (roi.contains("scoord")) for (nb::handle v : nb::cast<nb::list>(roi["scoord"])) r.data.push_back(nb::cast<float>(v));
    r.frame_of_reference_uid = sr_str(roi, "frame_of_reference_uid");
    r.ref_sop_class = sr_str(roi, "ref_sop_class_uid");
    r.ref_sop_instance = sr_str(roi, "ref_sop_instance_uid");
    return r;
}
nb::dict roi_dict(const dcmbase::sr::sr_roi& roi) {
    nb::dict d;
    d["graphic_type"] = sr_graphic_name(static_cast<std::uint8_t>(roi.graphic_type));
    d["is_3d"] = roi.is_3d;
    nb::list data; for (float v : roi.data) data.append(v); d["scoord"] = data;
    if (roi.is_3d) d["frame_of_reference_uid"] = roi.frame_of_reference_uid;
    else { d["ref_sop_class_uid"] = roi.ref_sop_class; d["ref_sop_instance_uid"] = roi.ref_sop_instance; }
    return d;
}

// write_measurement_report(document, output="") -> bytes (or None). Authors a typed
// TID 1500 Measurement Report (observer context + measurement groups w/ tracking,
// finding, finding sites, ROI, NUM measurements + method/derivation, qualitative
// evaluations) via dcmbase::sr::build_measurement_report.
nb::object write_measurement_report(nb::dict d, const std::string& output) {
    dcmbase::sr::measurement_report r;
    r.patient_name = sr_str(d, "patient_name"); r.patient_id = sr_str(d, "patient_id");
    r.study_uid = sr_str(d, "study_uid"); r.study_date = sr_str(d, "study_date");
    r.series_uid = sr_str(d, "series_uid"); r.sop_class_uid = sr_str(d, "sop_class_uid");
    r.completion_flag = sr_str(d, "completion_flag"); r.verification_flag = sr_str(d, "verification_flag");
    r.language = sr_str(d, "language");
    if (d.contains("observer")) {
        nb::dict o = nb::cast<nb::dict>(d["observer"]);
        r.observer_is_device = (sr_str(o, "type") != "person");
        r.person_observer_name = sr_str(o, "name");
        r.device_observer_uid = sr_str(o, "uid");
    }
    if (d.contains("procedure_reported")) r.procedure_reported = sr_code(d["procedure_reported"]);
    if (d.contains("image_library")) for (nb::handle ih : nb::cast<nb::list>(d["image_library"])) {
        nb::dict im = nb::cast<nb::dict>(ih);
        r.image_library.push_back({sr_str(im, "sop_class_uid"), sr_str(im, "sop_instance_uid")});
    }
    if (d.contains("groups")) for (nb::handle gh : nb::cast<nb::list>(d["groups"])) {
        nb::dict g = nb::cast<nb::dict>(gh);
        dcmbase::sr::sr_measurement_group mg;
        mg.tracking_id = sr_str(g, "tracking_id"); mg.tracking_uid = sr_str(g, "tracking_uid");
        if (g.contains("finding")) mg.finding = sr_code(g["finding"]);
        if (g.contains("finding_sites"))
            for (nb::handle h : nb::cast<nb::list>(g["finding_sites"])) mg.finding_sites.push_back(fs_from(h));
        if (g.contains("roi")) { mg.has_roi = true; mg.roi = roi_from(nb::cast<nb::dict>(g["roi"])); }
        if (g.contains("measurements")) for (nb::handle mh : nb::cast<nb::list>(g["measurements"])) {
            nb::dict m = nb::cast<nb::dict>(mh);
            dcmbase::sr::sr_measurement mm;
            if (m.contains("name")) mm.name = sr_code(m["name"]);
            mm.value = m.contains("value") ? nb::cast<double>(m["value"]) : 0.0;
            if (m.contains("unit")) mm.unit = sr_code(m["unit"]);
            if (m.contains("method")) mm.method = sr_code(m["method"]);
            if (m.contains("derivation")) mm.derivation = sr_code(m["derivation"]);
            if (m.contains("finding_sites"))
                for (nb::handle h : nb::cast<nb::list>(m["finding_sites"])) mm.finding_sites.push_back(fs_from(h));
            if (m.contains("region")) { mm.has_region = true; mm.region = roi_from(nb::cast<nb::dict>(m["region"])); }
            mm.equivalent_meaning = sr_str(m, "equivalent_meaning");
            if (m.contains("rwvm")) {
                nb::dict rw = nb::cast<nb::dict>(m["rwvm"]);
                mm.rwvm.sop_class = sr_str(rw, "sop_class_uid");
                mm.rwvm.sop_instance = sr_str(rw, "sop_instance_uid");
            }
            mg.measurements.push_back(std::move(mm));
        }
        if (g.contains("qualitative_evaluations")) for (nb::handle qh : nb::cast<nb::list>(g["qualitative_evaluations"])) {
            nb::dict q = nb::cast<nb::dict>(qh);
            mg.qualitative_evaluations.push_back({sr_code(q["name"]), sr_code(q["value"])});
        }
        r.groups.push_back(std::move(mg));
    }
    return emit_seg(dcmbase::sr::build_measurement_report(r), output, "failed to build the measurement report");
}

// read_measurement_report(path) -> {patient/study + observer + groups:[...]}. Typed
// TID 1500 parse, round-tripping write_measurement_report (empty groups when not one).
nb::object read_measurement_report_py(const std::string& path) {
    rdr::decoded dec{std::span<const std::byte>{slurp(path)}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dcmbase::sr::measurement_report r = dcmbase::sr::read_measurement_report(dec.get());
    nb::dict out;
    out["patient_name"] = r.patient_name; out["patient_id"] = r.patient_id;
    out["study_uid"] = r.study_uid; out["study_date"] = r.study_date; out["series_uid"] = r.series_uid;
    out["language"] = r.language;
    nb::dict obs; obs["type"] = r.observer_is_device ? "device" : "person";
    obs["name"] = r.person_observer_name; obs["uid"] = r.device_observer_uid;
    out["observer"] = obs;
    if (!r.procedure_reported.value.empty()) out["procedure_reported"] = code_dict(r.procedure_reported);
    if (!r.image_library.empty()) {
        nb::list il;
        for (const auto& im : r.image_library) {
            nb::dict d; d["sop_class_uid"] = im.sop_class; d["sop_instance_uid"] = im.sop_instance; il.append(d);
        }
        out["image_library"] = il;
    }
    nb::list groups;
    for (const auto& g : r.groups) {
        nb::dict gd;
        gd["tracking_id"] = g.tracking_id; gd["tracking_uid"] = g.tracking_uid;
        if (!g.finding.value.empty()) gd["finding"] = code_dict(g.finding);
        nb::list fs; for (const auto& c : g.finding_sites) fs.append(fs_dict(c)); gd["finding_sites"] = fs;
        if (g.has_roi) gd["roi"] = roi_dict(g.roi);
        nb::list meas;
        for (const auto& m : g.measurements) {
            nb::dict md;
            md["name"] = code_dict(m.name); md["value"] = m.value; md["unit"] = code_dict(m.unit);
            if (!m.method.value.empty()) md["method"] = code_dict(m.method);
            if (!m.derivation.value.empty()) md["derivation"] = code_dict(m.derivation);
            nb::list mfs; for (const auto& c : m.finding_sites) mfs.append(fs_dict(c)); md["finding_sites"] = mfs;
            if (m.has_region) md["region"] = roi_dict(m.region);
            if (!m.equivalent_meaning.empty()) md["equivalent_meaning"] = m.equivalent_meaning;
            if (!m.rwvm.sop_instance.empty()) {
                nb::dict rw; rw["sop_class_uid"] = m.rwvm.sop_class; rw["sop_instance_uid"] = m.rwvm.sop_instance;
                md["rwvm"] = rw;
            }
            meas.append(md);
        }
        gd["measurements"] = meas;
        nb::list qe;
        for (const auto& [n, v] : g.qualitative_evaluations) {
            nb::dict q; q["name"] = code_dict(n); q["value"] = code_dict(v); qe.append(q);
        }
        gd["qualitative_evaluations"] = qe;
        groups.append(gd);
    }
    out["groups"] = groups;
    return out;
}

// ---- Key Object Selection (write_ko / read_ko) -----------------------------
// write_ko(document, output="") -> bytes (or None). Authors a KOS (PS3.3 KOS IOD /
// PS3.16 TID 2010) via dcmbase::ko::build. document: {patient_name, patient_id,
// study_uid, study_date, study_time, study_id, accession_number, title:{value,scheme,
// meaning}, references:[{study_uid, series_uid, sop_class_uid, sop_instance_uid}, …]}.
nb::object write_ko(nb::dict d, const std::string& output) {
    std::vector<dcmbase::ko::reference> refs;
    if (d.contains("references")) for (nb::handle rh : nb::cast<nb::list>(d["references"])) {
        nb::dict r = nb::cast<nb::dict>(rh);
        refs.push_back({sr_str(r, "study_uid"), sr_str(r, "series_uid"),
                        sr_str(r, "sop_class_uid"), sr_str(r, "sop_instance_uid")});
    }
    dcmbase::ko::title t;
    if (d.contains("title")) {
        nb::dict td = nb::cast<nb::dict>(d["title"]);
        t.code = sr_str(td, "value"); t.scheme = sr_str(td, "scheme"); t.meaning = sr_str(td, "meaning");
    }
    return emit_seg(dcmbase::ko::build(
        sr_str(d, "patient_name"), sr_str(d, "patient_id"), sr_str(d, "study_uid"),
        sr_str(d, "study_date"), sr_str(d, "study_time"), sr_str(d, "study_id"),
        sr_str(d, "accession_number"), refs, t),
        output, "failed to build the Key Object Selection (no references?)");
}

// ---- Microscopy Bulk Simple Annotations (write_ann) ------------------------
// write_ann(document, output="") -> bytes | None. Authors an ANN object via the
// native dcmbase::ann::build (the exact inverse of read_ann). document carries the
// identity + groups; each group's `annotations` is a list of flat coordinate lists,
// flattened here into the bulk coords + value-based Long Primitive Point Index List.
static dcmbase::ann::code ann_code(nb::handle h) {
    dcmbase::ann::code c;
    if (h && nb::isinstance<nb::dict>(h)) {
        nb::dict cd = nb::cast<nb::dict>(h);
        c.value = sr_str(cd, "value"); c.scheme = sr_str(cd, "scheme"); c.meaning = sr_str(cd, "meaning");
    }
    return c;
}

nb::object write_ann(nb::dict d, const std::string& output) {
    dcmbase::ann::document doc;
    doc.coordinate_type = sr_str(d, "coordinate_type").empty() ? "2D" : sr_str(d, "coordinate_type");
    const int dim = (doc.coordinate_type == "3D") ? 3 : 2;

    for (nb::handle gh : nb::cast<nb::list>(d["groups"])) {
        nb::dict g = nb::cast<nb::dict>(gh);
        dcmbase::ann::group gr;
        gr.number          = g.contains("number") ? nb::cast<int>(g["number"]) : 1;
        gr.uid             = sr_str(g, "uid");
        gr.label           = sr_str(g, "label");
        gr.generation_type = sr_str(g, "generation_type");
        gr.graphic_type    = sr_str(g, "graphic_type");
        gr.dimensionality  = dim;
        gr.property_category = ann_code(g.contains("property_category") ? g["property_category"] : nb::handle());
        gr.property_type     = ann_code(g.contains("property_type") ? g["property_type"] : nb::handle());
        const bool variable = (gr.graphic_type == "POLYLINE" || gr.graphic_type == "POLYGON");
        std::uint32_t pos = 1, count = 0;
        for (nb::handle ah : nb::cast<nb::list>(g["annotations"])) {
            if (variable) gr.index_list.push_back(pos);            // value-based 1-based offset
            for (nb::handle vh : nb::cast<nb::list>(ah)) { gr.coords.push_back(nb::cast<double>(vh)); ++pos; }
            ++count;
        }
        gr.num_annotations = count;
        if (g.contains("measurements")) for (nb::handle mh : nb::cast<nb::list>(g["measurements"])) {
            nb::dict m = nb::cast<nb::dict>(mh);
            dcmbase::ann::measurement me;
            me.name = ann_code(m.contains("name") ? m["name"] : nb::handle());
            me.unit = ann_code(m.contains("unit") ? m["unit"] : nb::handle());
            for (nb::handle vh : nb::cast<nb::list>(m["values"])) me.values.push_back(nb::cast<double>(vh));
            if (m.contains("annotation_index") && !m["annotation_index"].is_none())
                for (nb::handle ih : nb::cast<nb::list>(m["annotation_index"]))
                    me.annotation_index.push_back(nb::cast<std::uint32_t>(ih));
            gr.measurements.push_back(std::move(me));
        }
        doc.groups.push_back(std::move(gr));
    }

    dcmbase::ann::identity id;
    id.patient_name = sr_str(d, "patient_name");   id.patient_id = sr_str(d, "patient_id");
    id.study_uid = sr_str(d, "study_uid");         id.study_date = sr_str(d, "study_date");
    id.study_time = sr_str(d, "study_time");       id.study_id = sr_str(d, "study_id");
    id.accession_number = sr_str(d, "accession_number"); id.series_uid = sr_str(d, "series_uid");
    id.series_number = sr_str(d, "series_number"); id.sop_uid = sr_str(d, "sop_uid");
    id.instance_number = sr_str(d, "instance_number");
    id.frame_of_reference_uid = sr_str(d, "frame_of_reference_uid");
    id.manufacturer = sr_str(d, "manufacturer");   id.manufacturer_model = sr_str(d, "manufacturer_model");
    id.software_versions = sr_str(d, "software_versions"); id.device_serial = sr_str(d, "device_serial");
    if (d.contains("references")) for (nb::handle rh : nb::cast<nb::list>(d["references"])) {
        nb::dict r = nb::cast<nb::dict>(rh);
        id.references.push_back({sr_str(r, "sop_class"), sr_str(r, "sop_instance")});
    }
    return emit_seg(dcmbase::ann::build(doc, id), output, "failed to build the Bulk Annotations");
}

// read_ko(path) -> {patient/study, title, references:[{sop_class_uid, sop_instance_uid}]}
// | None (when not a KOS). References are the IMAGE content items of the parsed tree.
nb::object read_ko(const std::string& path) {
    rdr::decoded dec{std::span<const std::byte>{slurp(path)}};
    if (!dec) throw std::runtime_error("decode failed (not a decodable DICOM): " + path);
    const dicom_info_t* info = dec.get();
    if (cstr(info->sop_class_uid) != std::string(UID_KeyObjectSelectionDocumentStorage)) return nb::none();
    nb::dict out;
    out["patient_name"] = cstr(info->patient_name); out["patient_id"] = cstr(info->patient_id);
    out["study_uid"] = cstr(info->study_instance_uid); out["series_uid"] = cstr(info->series_instance_uid);
    const sr_document_t& sr = info->sr;
    int root = -1;
    for (std::uint16_t i = 0; i < sr.count; ++i) if (sr.nodes[i].parent_index < 0) { root = i; break; }
    if (root >= 0) {
        nb::dict t;
        t["value"] = cstr(sr.nodes[root].concept_value);
        t["scheme"] = cstr(sr.nodes[root].concept_scheme);
        t["meaning"] = cstr(sr.nodes[root].concept_meaning);
        out["title"] = t;
    }
    nb::list refs;
    for (std::uint16_t i = 0; i < sr.count; ++i) if (sr.nodes[i].value_type == SR_VT_IMAGE) {
        nb::dict r;
        r["sop_class_uid"] = cstr(sr.nodes[i].ref.sop_class);
        r["sop_instance_uid"] = cstr(sr.nodes[i].ref.sop_instance);
        refs.append(r);
    }
    out["references"] = refs;
    return out;
}

// ---- Grayscale Softcopy Presentation State (write_pr) -----------------------
// write_pr(document, output="") -> bytes (or None). Authors a GSPS via dcmbase's
// dcm_ps_export engine. document: {patient_name, patient_id, study_uid, study_date,
// content_label, content_description, content_creator, presentation_lut_shape
// ("IDENTITY"/"INVERSE"), rotation (0/90/180/270), h_flip, references:[{series_uid,
// sop_class_uid, sop_instance_uid, frame_numbers}], voi_luts:[{window_center,
// window_width, function, explanation}], displayed_areas:[{tlhc:[x,y], brhc:[x,y],
// size_mode, magnification, pixel_spacing:[x,y]}], graphic_layers:[{name, order,
// description, cielab:[L,a,b]}], graphic_annotations:[{layer, texts:[...], graphics:[...]}]}.
nb::object write_pr(nb::dict d, const std::string& output) {
    auto p2 = [](nb::handle h, int idx) -> double {  // numeric element idx of a list
        nb::list l = nb::cast<nb::list>(h); return nb::cast<double>(l[idx]);
    };
    ps_export_begin(sr_str(d, "patient_name").c_str(), sr_str(d, "patient_id").c_str(),
                    sr_str(d, "study_uid").c_str(), sr_str(d, "study_date").c_str(),
                    sr_str(d, "content_label").c_str(), sr_str(d, "content_description").c_str(),
                    sr_str(d, "content_creator").c_str());
    ps_export_set_presentation_lut(sr_str(d, "presentation_lut_shape") == "INVERSE" ? 1 : 0);
    int rot = d.contains("rotation") ? nb::cast<int>(d["rotation"]) : 0;
    bool hflip = d.contains("h_flip") && nb::cast<bool>(d["h_flip"]);
    if (rot || hflip) ps_export_set_spatial(rot, hflip ? 1 : 0);

    // PS type / SOP class: GSPS | COLOR (11.2) | PSEUDO_COLOR (11.3) | XAXRF (11.5) | ADVANCED_BLENDING (11.8).
    const std::string kind = sr_str(d, "ps_type");
    ps_export_set_sop_class(kind == "COLOR" ? 2 : kind == "PSEUDO_COLOR" ? 3 : kind == "XAXRF" ? 5 :
                            kind == "ADVANCED_BLENDING" ? 6 : 1);
    // Palette Color LUT (Pseudo-Color): three equal-length 16-bit channels.
    if (d.contains("palette")) {
        nb::dict pal = nb::cast<nb::dict>(d["palette"]);
        auto chan = [&](const char* k) {
            std::vector<std::uint16_t> v;
            for (nb::handle h : nb::cast<nb::list>(pal[k])) v.push_back((std::uint16_t)nb::cast<int>(h));
            return v;
        };
        std::vector<std::uint16_t> red = chan("red"), green = chan("green"), blue = chan("blue");
        if (!red.empty() && red.size() == green.size() && green.size() == blue.size()) {
            // entry count IS the channel length — the engine memcpy's entries*2 bytes
            // from these buffers, so deriving it here (not trusting a user field) can't over-read.
            std::uint32_t entries = (std::uint32_t)red.size();
            std::uint16_t first = pal.contains("first_mapped") ? (std::uint16_t)nb::cast<int>(pal["first_mapped"]) : 0;
            ps_export_set_palette_lut(entries, first, red.data(), green.data(), blue.data());
        }
    }
    // ICC Profile + Color Space (Color).
    if (d.contains("icc_profile")) {
        nb::bytes icc = nb::cast<nb::bytes>(d["icc_profile"]);
        ps_export_set_icc_profile(reinterpret_cast<const std::uint8_t*>(icc.c_str()), (std::uint32_t)icc.size());
    }
    if (d.contains("color_space")) ps_export_set_color_space(sr_str(d, "color_space").c_str());
    // Mask Subtraction (XA/XRF).
    if (d.contains("mask")) {
        nb::dict m = nb::cast<nb::dict>(d["mask"]);
        const std::string op = sr_str(m, "operation");
        int operation = op == "TID" ? 2 : op == "REV_TID" ? 3 : 1;   // PS_MASK_AVG_SUB=1/TID=2/REV_TID=3
        std::vector<std::uint16_t> frames;
        if (m.contains("mask_frames")) for (nb::handle h : nb::cast<nb::list>(m["mask_frames"]))
            frames.push_back((std::uint16_t)nb::cast<int>(h));
        std::uint16_t range[2] = {0, 0}; bool has_range = m.contains("applicable_range");
        if (has_range) { range[0] = (std::uint16_t)p2(m["applicable_range"], 0); range[1] = (std::uint16_t)p2(m["applicable_range"], 1); }
        float shift[2] = {0, 0}; bool has_shift = m.contains("sub_pixel_shift");
        if (has_shift) { shift[0] = (float)p2(m["sub_pixel_shift"], 0); shift[1] = (float)p2(m["sub_pixel_shift"], 1); }
        bool has_tid = m.contains("tid_offset");
        ps_export_set_mask(operation, frames.empty() ? nullptr : frames.data(), (std::uint32_t)frames.size(),
                           has_range ? range : nullptr, has_shift ? shift : nullptr,
                           has_tid ? nb::cast<int>(m["tid_offset"]) : 0, has_tid ? 1 : 0);
    }
    // Advanced Blending (XA/XRF's sibling): N inputs (refs + palette) + M blending-display items.
    if (d.contains("blending")) for (nb::handle bh : nb::cast<nb::list>(d["blending"])) {
        nb::dict bi = nb::cast<nb::dict>(bh);
        int innum = bi.contains("input_number") ? nb::cast<int>(bi["input_number"]) : 1;
        std::int32_t idx = ps_export_ab_add_input(innum, sr_str(bi, "study_uid").c_str(),
                                                  sr_str(bi, "series_uid").c_str());
        if (bi.contains("references")) for (nb::handle rh : nb::cast<nb::list>(bi["references"])) {
            nb::dict r = nb::cast<nb::dict>(rh);
            ps_export_ab_input_add_reference(idx, sr_str(r, "sop_class_uid").c_str(),
                                             sr_str(r, "sop_instance_uid").c_str());
        }
        if (bi.contains("palette")) {
            nb::dict pal = nb::cast<nb::dict>(bi["palette"]);
            auto chan = [&](const char* k) {
                std::vector<std::uint16_t> v;
                for (nb::handle h : nb::cast<nb::list>(pal[k])) v.push_back((std::uint16_t)nb::cast<int>(h));
                return v;
            };
            std::vector<std::uint16_t> red = chan("red"), green = chan("green"), blue = chan("blue");
            if (!red.empty() && red.size() == green.size() && green.size() == blue.size()) {
                std::uint16_t first = pal.contains("first_mapped") ? (std::uint16_t)nb::cast<int>(pal["first_mapped"]) : 0;
                ps_export_ab_input_set_palette(idx, (std::uint32_t)red.size(), first,
                                               red.data(), green.data(), blue.data());
            }
        }
    }
    if (d.contains("blending_display")) for (nb::handle dh : nb::cast<nb::list>(d["blending_display"])) {
        nb::dict bd = nb::cast<nb::dict>(dh);
        int mode = sr_str(bd, "mode") == "FOREGROUND" ? 1 : 0;
        bool has_op = bd.contains("relative_opacity");
        std::vector<std::uint16_t> ins;
        if (bd.contains("inputs")) for (nb::handle h : nb::cast<nb::list>(bd["inputs"]))
            ins.push_back((std::uint16_t)nb::cast<int>(h));
        ps_export_ab_add_display(mode, has_op ? 1 : 0, has_op ? nb::cast<float>(bd["relative_opacity"]) : 0.0f,
                                 ins.empty() ? nullptr : ins.data(), (std::uint32_t)ins.size());
    }

    if (d.contains("references")) for (nb::handle rh : nb::cast<nb::list>(d["references"])) {
        nb::dict r = nb::cast<nb::dict>(rh);
        ps_export_add_reference(sr_str(r, "series_uid").c_str(), sr_str(r, "sop_class_uid").c_str(),
                                sr_str(r, "sop_instance_uid").c_str(), sr_str(r, "frame_numbers").c_str());
    }
    if (d.contains("voi_luts")) for (nb::handle vh : nb::cast<nb::list>(d["voi_luts"])) {
        nb::dict v = nb::cast<nb::dict>(vh);
        const std::string fn = sr_str(v, "function");
        int func = fn == "LINEAR_EXACT" ? 1 : fn == "SIGMOID" ? 2 : 0;
        ps_export_add_voi(nb::cast<double>(v["window_center"]), nb::cast<double>(v["window_width"]),
                          func, sr_str(v, "explanation").c_str());
    }
    if (d.contains("displayed_areas")) for (nb::handle dh : nb::cast<nb::list>(d["displayed_areas"])) {
        nb::dict da = nb::cast<nb::dict>(dh);
        const std::string sm = sr_str(da, "size_mode");
        int mode = sm == "TRUE SIZE" ? 1 : sm == "MAGNIFY" ? 2 : 0;
        float mag = da.contains("magnification") ? nb::cast<float>(da["magnification"]) : 0.0f;
        float psx = 0, psy = 0;
        if (da.contains("pixel_spacing")) { psx = (float)p2(da["pixel_spacing"], 0); psy = (float)p2(da["pixel_spacing"], 1); }
        ps_export_add_displayed_area((int32_t)p2(da["tlhc"], 0), (int32_t)p2(da["tlhc"], 1),
                                     (int32_t)p2(da["brhc"], 0), (int32_t)p2(da["brhc"], 1),
                                     mode, mag, psx, psy);
    }
    if (d.contains("graphic_layers")) for (nb::handle lh : nb::cast<nb::list>(d["graphic_layers"])) {
        nb::dict l = nb::cast<nb::dict>(lh);
        int has_color = l.contains("cielab") ? 1 : 0;
        std::uint16_t c0 = 0, c1 = 0, c2 = 0;
        if (has_color) { c0 = (std::uint16_t)p2(l["cielab"], 0); c1 = (std::uint16_t)p2(l["cielab"], 1); c2 = (std::uint16_t)p2(l["cielab"], 2); }
        ps_export_add_layer(sr_str(l, "name").c_str(), l.contains("order") ? nb::cast<int>(l["order"]) : 1,
                            sr_str(l, "description").c_str(), has_color, c0, c1, c2);
    }
    if (d.contains("graphic_annotations")) for (nb::handle ah : nb::cast<nb::list>(d["graphic_annotations"])) {
        nb::dict a = nb::cast<nb::dict>(ah);
        std::int32_t ann = ps_export_add_annotation(sr_str(a, "layer").c_str());
        if (a.contains("texts")) for (nb::handle th : nb::cast<nb::list>(a["texts"])) {
            nb::dict t = nb::cast<nb::dict>(th);
            const std::string ju = sr_str(t, "justification");
            int just = ju == "CENTER" ? 1 : ju == "RIGHT" ? 2 : 0;
            int units = sr_str(t, "units") == "DISPLAY" ? 1 : 0;
            int has_bbox = t.contains("bounding_box") ? 1 : 0, has_anchor = t.contains("anchor") ? 1 : 0;
            float bb[4] = {0, 0, 0, 0}, an[2] = {0, 0};
            if (has_bbox) for (int i = 0; i < 4; ++i) bb[i] = (float)p2(t["bounding_box"], i);
            if (has_anchor) { an[0] = (float)p2(t["anchor"], 0); an[1] = (float)p2(t["anchor"], 1); }
            ps_export_annotation_add_text(ann, sr_str(t, "text").c_str(), has_bbox, bb[0], bb[1], bb[2], bb[3],
                                          just, has_anchor, an[0], an[1], units);
        }
        if (a.contains("graphics")) for (nb::handle gh : nb::cast<nb::list>(a["graphics"])) {
            nb::dict g = nb::cast<nb::dict>(gh);
            const std::string gt = sr_str(g, "graphic_type");
            int type = gt == "POLYLINE" ? 1 : gt == "INTERPOLATED" ? 2 : gt == "CIRCLE" ? 3 : gt == "ELLIPSE" ? 4 : 0;
            int units = sr_str(g, "units") == "DISPLAY" ? 1 : 0;
            int filled = g.contains("filled") && nb::cast<bool>(g["filled"]);
            std::vector<float> pts;
            if (g.contains("points")) for (nb::handle v : nb::cast<nb::list>(g["points"])) pts.push_back(nb::cast<float>(v));
            ps_export_annotation_add_graphic(ann, type, (std::uint16_t)(pts.size() / 2),
                                             pts.empty() ? nullptr : pts.data(), units, filled);
        }
    }

    std::uint32_t out_size = 0;
    const std::uint8_t* buf = ps_export_finalize(&out_size);
    std::vector<std::byte> bytes;
    if (buf && out_size)
        bytes.assign(reinterpret_cast<const std::byte*>(buf), reinterpret_cast<const std::byte*>(buf) + out_size);
    ps_export_free();
    return emit_seg(bytes, output, "failed to build the presentation state");
}

// write_legacy_converted(reference_paths, options, output="") -> bytes | None.
// Folds a classic single-frame CT/MR/PET series into ONE Legacy Converted
// Enhanced multi-frame object via dcmbase::legacy::convert (the
// `legacy` capability). Source buffers are kept alive across the call (the
// converter borrows their pixels/attributes).
nb::object write_legacy_converted(const std::vector<std::string>& reference_paths,
                                  nb::dict options, const std::string& output) {
    std::vector<std::vector<std::byte>> bufs;
    bufs.reserve(reference_paths.size());
    for (const auto& p : reference_paths) bufs.push_back(slurp(p));
    std::vector<std::span<const std::byte>> spans;
    spans.reserve(bufs.size());
    for (const auto& bb : bufs) spans.emplace_back(bb);

    dcmbase::legacy_converted::options o;
    auto gs = [&](const char* k) {
        return options.contains(k) ? nb::cast<std::string>(options[k]) : std::string{};
    };
    o.series_instance_uid = gs("series_instance_uid");
    o.sop_instance_uid    = gs("sop_instance_uid");
    if (options.contains("series_number"))   o.series_number   = nb::cast<int>(options["series_number"]);
    if (options.contains("instance_number")) o.instance_number = nb::cast<int>(options["instance_number"]);
    o.manufacturer        = gs("manufacturer");
    o.model_name          = gs("model_name");
    o.device_serial       = gs("device_serial");
    o.software_versions   = gs("software_versions");

    auto res = dcmbase::legacy_converted::convert(spans, o);
    if (!res)
        throw std::runtime_error(std::string("legacy conversion failed: ") +
                                 dcmbase::legacy_converted::describe(res.error()));
    return emit_seg(*res, output, "legacy conversion produced no output");
}

// read_ann(path) -> {coordinate_type, groups:[{...}]} | None. Microscopy Bulk
// Simple Annotations reader (dcmbase::ann::read). Bulk point coordinates are
// returned as raw float64 bytes; pydcm.read_ann reshapes/splits them with NumPy.
nb::object read_ann(const std::string& path) {
    const auto doc = dcmbase::ann::read(std::span<const std::byte>{slurp(path)});
    if (!doc) return nb::none();
    auto code_d = [](const dcmbase::ann::code& c) {
        nb::dict d; d["value"] = c.value; d["scheme"] = c.scheme; d["meaning"] = c.meaning; return d;
    };
    nb::list groups;
    for (const auto& g : doc->groups) {
        nb::dict gd;
        gd["number"] = g.number;
        gd["uid"] = g.uid;
        gd["label"] = g.label;
        gd["generation_type"] = g.generation_type;
        gd["property_category"] = code_d(g.property_category);
        gd["property_type"] = code_d(g.property_type);
        gd["graphic_type"] = g.graphic_type;
        gd["dimensionality"] = g.dimensionality;
        gd["num_annotations"] = g.num_annotations;
        gd["coords"] = nb::bytes(reinterpret_cast<const char*>(g.coords.data()),
                                 g.coords.size() * sizeof(double));
        nb::list idx;
        for (auto v : g.index_list) idx.append(v);
        gd["index_list"] = idx;
        nb::list meas;
        for (const auto& m : g.measurements) {
            nb::dict md;
            md["name"] = code_d(m.name);
            md["unit"] = code_d(m.unit);
            md["values"] = nb::bytes(reinterpret_cast<const char*>(m.values.data()),
                                     m.values.size() * sizeof(double));
            nb::list ai;
            for (auto v : m.annotation_index) ai.append(v);
            md["annotation_index"] = ai;
            meas.append(md);
        }
        gd["measurements"] = meas;
        groups.append(gd);
    }
    nb::dict out;
    out["coordinate_type"] = doc->coordinate_type;
    out["groups"] = groups;
    return out;
}

nb::ndarray<nb::numpy> uint8_image_array(std::vector<std::uint8_t>&& data,
                                         std::size_t h, std::size_t w,
                                         std::size_t ch) {
    auto* out = new std::vector<std::uint8_t>(std::move(data));
    std::size_t shape[3] = {h, w, ch};
    if (out->empty()) {
        shape[0] = 0;
        shape[1] = 0;
    }
    nb::capsule owner(out, [](void* p) noexcept {
        delete static_cast<std::vector<std::uint8_t>*>(p);
    });
    return nb::ndarray<nb::numpy>(
        out->data(), 3, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
}

bool checked_size_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        return false;
    out = a * b;
    return true;
}

std::pair<std::uint32_t, std::uint32_t> checked_tile_pair(nb::handle item) {
    auto t = nb::cast<nb::tuple>(item);
    if (t.size() != 2)
        throw std::invalid_argument("tile must be a (tile_x, tile_y) pair");
    const auto tx = nb::cast<std::int64_t>(t[0]);
    const auto ty = nb::cast<std::int64_t>(t[1]);
    constexpr auto max_u32 = static_cast<std::int64_t>(
        std::numeric_limits<std::uint32_t>::max());
    if (tx < 0 || ty < 0 || tx > max_u32 || ty > max_u32)
        throw std::out_of_range("tile coordinates must fit uint32");
    return {static_cast<std::uint32_t>(tx), static_cast<std::uint32_t>(ty)};
}

nb::ndarray<nb::numpy>
wsi_range_table_array(std::vector<dcmbase::wsi::encoded_tile_range> ranges) {
    auto* out = new std::vector<std::uint64_t>();
    out->reserve(ranges.size() * 6U);
    for (const auto& r : ranges) {
        out->push_back(r.source_index);
        out->push_back(r.frame_index);
        out->push_back(r.tile_x);
        out->push_back(r.tile_y);
        out->push_back(r.offset);
        out->push_back(r.length);
    }
    std::size_t shape[2] = {ranges.size(), 6};
    nb::capsule owner(out, [](void* p) noexcept {
        delete static_cast<std::vector<std::uint64_t>*>(p);
    });
    return nb::ndarray<nb::numpy>(
        out->data(), 2, shape, owner, nullptr,
        nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 64, 1});
}

// Author a VL WSM pyramid from already-tiled (TILED_FULL) per-level RGB buffers — the
// numpy-friendly authoring path (pydcm.wsi.write_slide tiles the arrays, this wires the
// engine). Each tile buffer is total_rows*total_cols padded to whole tiles, tile-major.
nb::list wsi_write_pyramid_py(std::vector<nb::bytes> tile_bufs,
                              std::vector<int> cols, std::vector<int> rows, std::vector<int> tiles,
                              double mpp, const std::string& patient_id, const std::string& patient_name,
                              const std::string& study_uid, const std::string& container_id,
                              const std::string& specimen_id, const std::string& ts, int quality) {
    const std::size_t n = tile_bufs.size();
    if (n == 0 || cols.size() != n || rows.size() != n || tiles.size() != n)
        throw std::runtime_error("wsi.write_slide: mismatched / empty level arrays");
    dcmbase::wsi::write_meta meta;
    meta.patient_id = patient_id; meta.patient_name = patient_name;
    meta.study_instance_uid = study_uid;
    meta.container_id = container_id; meta.specimen_id = specimen_id;
    meta.mpp_x = meta.mpp_y = mpp;

    std::vector<dcmbase::wsi::write_level> levels(n);
    for (std::size_t i = 0; i < n; ++i) {
        levels[i].tiles = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(tile_bufs[i].c_str()), tile_bufs[i].size());
        levels[i].total_cols = static_cast<std::uint32_t>(cols[i]);
        levels[i].total_rows = static_cast<std::uint32_t>(rows[i]);
        levels[i].tile_cols = levels[i].tile_rows = static_cast<std::uint32_t>(tiles[i]);
    }
    auto r = dcmbase::wsi::build_pyramid(meta, levels, ts, quality);
    if (!r) throw std::runtime_error("wsi.write_slide: " + r.error());
    nb::list out;
    for (const auto& inst : *r)
        out.append(nb::bytes(reinterpret_cast<const char*>(inst.data()), inst.size()));
    return out;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "pydcm native core — zero-copy DICOM pixel decode via dcmbase::render (HU on rescale=True)";
    m.def("wsi_write_pyramid", &wsi_write_pyramid_py,
          "Author a VL WSM pyramid from per-level TILED_FULL RGB buffers → list[Part-10 bytes].");

    // ---- Whole-slide pyramid reader (dcmbase::wsi::slide) — backs pydcm.wsi ----
    nb::class_<dcmbase::wsi::slide>(m, "WsiSlide")
        .def("open_error", &dcmbase::wsi::slide::open_error)
        .def("level_count", &dcmbase::wsi::slide::level_count)
        .def("level_dimensions", [](const dcmbase::wsi::slide& s, int level) {
            auto [c, r] = s.level_dimensions(level); return nb::make_tuple(c, r);
        }, nb::arg("level"))
        .def("level_tile_dimensions", [](const dcmbase::wsi::slide& s, int level) {
            auto [c, r] = s.level_tile_dimensions(level); return nb::make_tuple(c, r);
        }, nb::arg("level"))
        .def("level_tile_counts", [](const dcmbase::wsi::slide& s, int level) {
            auto [c, r] = s.level_tile_counts(level); return nb::make_tuple(c, r);
        }, nb::arg("level"))
        .def("level_descriptor", [](const dcmbase::wsi::slide& s, int level) -> nb::object {
            const auto desc = s.level_descriptor(level);
            if (!desc) return nb::none();
            nb::dict d;
            d["width"] = desc->width;
            d["height"] = desc->height;
            d["tile_width"] = desc->tile_width;
            d["tile_height"] = desc->tile_height;
            d["tile_count_x"] = desc->tile_count_x;
            d["tile_count_y"] = desc->tile_count_y;
            d["frame_count"] = desc->frame_count;
            d["downsample"] = desc->downsample;
            d["mpp_x"] = desc->mpp_x;
            d["mpp_y"] = desc->mpp_y;
            d["imaged_volume_width"] = desc->imaged_volume_width;
            d["imaged_volume_height"] = desc->imaged_volume_height;
            d["transfer_syntax"] = desc->transfer_syntax;
            d["transfer_syntax_uid"] = nb::str(desc->transfer_syntax_uid.c_str());
            d["dimension_organization"] = desc->dimension_organization;
            d["dimension_organization_type"] =
                nb::str(desc->dimension_organization_type.c_str());
            d["image_type_flags"] = desc->image_type_flags;
            d["samples_per_pixel"] = desc->samples_per_pixel;
            d["bits_allocated"] = desc->bits_allocated;
            d["bits_stored"] = desc->bits_stored;
            d["sparse"] = desc->sparse;
            d["has_extended_offset_table"] = desc->has_extended_offset_table;
            d["has_icc_profile"] = desc->has_icc_profile;
            return d;
        }, nb::arg("level"))
        .def("tile_exists", &dcmbase::wsi::slide::tile_exists,
             nb::arg("level"), nb::arg("tile_x"), nb::arg("tile_y"))
        .def("icc_transform_available", [](const dcmbase::wsi::slide&) {
            return dcmbase::wsi::slide::icc_transform_available();
        })
        .def("level_frame_count", &dcmbase::wsi::slide::level_frame_count,
             nb::arg("level"))
        .def("level_frame_tile", [](const dcmbase::wsi::slide& s, int level,
                                    std::uint32_t frame_index) -> nb::object {
            const auto tile = s.level_frame_tile(level, frame_index);
            if (!tile) return nb::none();
            return nb::make_tuple(tile->first, tile->second);
        }, nb::arg("level"), nb::arg("frame_index"))
        .def("level_source_paths", [](const dcmbase::wsi::slide& s, int level) {
            nb::list out;
            for (const auto& path : s.level_source_paths(level))
                out.append(nb::str(path.c_str()));
            return out;
        }, nb::arg("level"))
        .def("level_tile_range", [](const dcmbase::wsi::slide& s, int level,
                                    std::uint32_t tile_x,
                                    std::uint32_t tile_y) -> nb::object {
            const auto r = s.level_tile_range(level, tile_x, tile_y);
            if (!r) return nb::none();
            return nb::make_tuple(r->source_index, r->frame_index,
                                  r->tile_x, r->tile_y, r->offset, r->length);
        }, nb::arg("level"), nb::arg("tile_x"), nb::arg("tile_y"))
        .def("level_frame_range", [](const dcmbase::wsi::slide& s, int level,
                                     std::uint32_t frame_index) -> nb::object {
            const auto r = s.level_frame_range(level, frame_index);
            if (!r) return nb::none();
            return nb::make_tuple(r->source_index, r->frame_index,
                                  r->tile_x, r->tile_y, r->offset, r->length);
        }, nb::arg("level"), nb::arg("frame_index"))
        .def("level_tile_range_grid", [](const dcmbase::wsi::slide& s, int level) {
            return wsi_range_table_array(s.level_tile_range_grid(level));
        }, nb::arg("level"))
        .def("level_tile_ranges", [](const dcmbase::wsi::slide& s, int level) {
            return wsi_range_table_array(s.level_tile_ranges(level));
        }, nb::arg("level"))
        .def("level_concatenation", [](const dcmbase::wsi::slide& s, int level) -> nb::object {
            const auto c = s.level_concatenation(level);
            if (!c) return nb::none();
            nb::dict d;
            d["uid"] = nb::str(c->uid.c_str());
            d["in_number"] = c->in_number;
            d["total_number"] = c->total_number;
            d["frame_offset_number"] = c->frame_offset_number;
            d["source_sop_instance_uid"] = nb::str(c->source_sop_instance_uid.c_str());
            return d;
        }, nb::arg("level"))
        .def("level_downsample", &dcmbase::wsi::slide::level_downsample, nb::arg("level"))
        .def("best_level_for_downsample", &dcmbase::wsi::slide::best_level_for_downsample,
             nb::arg("downsample"))
        .def("tile_cache_capacity", &dcmbase::wsi::slide::tile_cache_capacity)
        .def("set_tile_cache_capacity", &dcmbase::wsi::slide::set_tile_cache_capacity,
             nb::arg("bytes"))
        .def("associated_image_names", &dcmbase::wsi::slide::associated_image_names)
        .def("associated_image_dimensions", [](const dcmbase::wsi::slide& s, const std::string& name) {
            auto [c, r] = s.associated_image_dimensions(name); return nb::make_tuple(c, r);
        }, nb::arg("name"))
        .def("read_associated_image", [](const dcmbase::wsi::slide& s, const std::string& name,
                                         bool rgba) {
            auto [w, h] = s.associated_image_dimensions(name);
            auto* out = new std::vector<std::uint8_t>(s.read_associated_image(name, rgba));
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty()) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("name"), nb::arg("rgba") = true)
        .def("read_associated_image_srgb", [](const dcmbase::wsi::slide& s, const std::string& name,
                                              bool rgba) {
            auto [w, h] = s.associated_image_dimensions(name);
            auto* out = new std::vector<std::uint8_t>(s.read_associated_image_srgb(name, rgba));
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty()) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("name"), nb::arg("rgba") = true)
        .def("color_profile", [](const dcmbase::wsi::slide& s) -> nb::object {
            const auto profile = s.color_profile();
            if (profile.empty()) return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(profile.data()), profile.size());
        })
        .def("level_color_profile", [](const dcmbase::wsi::slide& s, int level) -> nb::object {
            const auto profile = s.level_color_profile(level);
            if (profile.empty()) return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(profile.data()), profile.size());
        }, nb::arg("level"))
        .def("associated_image_color_profile", [](const dcmbase::wsi::slide& s,
                                                  const std::string& name) -> nb::object {
            const auto profile = s.associated_image_color_profile(name);
            if (profile.empty()) return nb::none();
            return nb::bytes(reinterpret_cast<const char*>(profile.data()), profile.size());
        }, nb::arg("name"))
        .def("properties", [](const dcmbase::wsi::slide& s) {
            nb::dict d;
            for (const auto& [key, value] : s.properties())
                d[nb::str(key.c_str())] = nb::str(value.c_str());
            return d;
        })
        .def("read_region", [](const dcmbase::wsi::slide& s, std::int64_t x, std::int64_t y,
                               int level, std::uint32_t w, std::uint32_t h, bool rgba) {
            auto* out = new std::vector<std::uint8_t>(s.read_region(x, y, level, w, h, rgba));
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty() && w != 0 && h != 0) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("x"), nb::arg("y"), nb::arg("level"), nb::arg("w"), nb::arg("h"), nb::arg("rgba") = true)
        .def("read_region_srgb", [](const dcmbase::wsi::slide& s, std::int64_t x, std::int64_t y,
                                    int level, std::uint32_t w, std::uint32_t h, bool rgba) {
            auto* out = new std::vector<std::uint8_t>(s.read_region_srgb(x, y, level, w, h, rgba));
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty() && w != 0 && h != 0) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("x"), nb::arg("y"), nb::arg("level"), nb::arg("w"), nb::arg("h"), nb::arg("rgba") = true)
        .def("read_tile", [](const dcmbase::wsi::slide& s, int level, std::uint32_t tile_x,
                             std::uint32_t tile_y, bool rgba, bool fill_missing) {
            auto* out = new std::vector<std::uint8_t>(
                s.read_tile(level, tile_x, tile_y, rgba, fill_missing));
            auto [w, h] = s.level_tile_dimensions(level);
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty()) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("level"), nb::arg("tile_x"), nb::arg("tile_y"),
           nb::arg("rgba") = true, nb::arg("fill_missing") = false)
        .def("read_tile_srgb", [](const dcmbase::wsi::slide& s, int level, std::uint32_t tile_x,
                                  std::uint32_t tile_y, bool rgba, bool fill_missing) {
            auto* out = new std::vector<std::uint8_t>(
                s.read_tile_srgb(level, tile_x, tile_y, rgba, fill_missing));
            auto [w, h] = s.level_tile_dimensions(level);
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t shape[3] = {h, w, ch};
            if (out->empty()) {
                shape[0] = 0;
                shape[1] = 0;
            }
            nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
            return nb::ndarray<nb::numpy>(out->data(), 3, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("level"), nb::arg("tile_x"), nb::arg("tile_y"),
           nb::arg("rgba") = true, nb::arg("fill_missing") = false)
        .def("read_tiles", [](const dcmbase::wsi::slide& s, int level, nb::list tiles,
                              bool rgba, bool fill_missing) {
            auto [w, h] = s.level_tile_dimensions(level);
            const std::size_t ch = rgba ? 4 : 3;
            nb::list out;
            for (nb::handle item : tiles) {
                auto [tile_x, tile_y] = checked_tile_pair(item);
                out.append(uint8_image_array(
                    s.read_tile(level, tile_x, tile_y, rgba, fill_missing), h, w, ch));
            }
            return out;
        }, nb::arg("level"), nb::arg("tiles"),
           nb::arg("rgba") = true, nb::arg("fill_missing") = false)
        .def("read_tile_stack", [](const dcmbase::wsi::slide& s, int level, nb::list tiles,
                                   bool rgba, bool fill_missing) {
            auto [w, h] = s.level_tile_dimensions(level);
            if ((w == 0 || h == 0) && nb::len(tiles) != 0)
                throw std::invalid_argument("invalid WSI level or zero tile dimensions");
            const std::size_t n = nb::len(tiles);
            const std::size_t ch = rgba ? 4 : 3;
            std::size_t pixels_per_tile = 0;
            std::size_t bytes_per_tile = 0;
            std::size_t total_bytes = 0;
            if (!checked_size_mul(static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                                  pixels_per_tile) ||
                !checked_size_mul(pixels_per_tile, ch, bytes_per_tile) ||
                !checked_size_mul(n, bytes_per_tile, total_bytes))
                throw std::overflow_error("read_tile_stack output is too large");

            auto* out = new std::vector<std::uint8_t>(total_bytes);
            try {
                std::size_t i = 0;
                for (nb::handle item : tiles) {
                    auto [tile_x, tile_y] = checked_tile_pair(item);
                    auto tile = s.read_tile(level, tile_x, tile_y, rgba, fill_missing);
                    if (tile.size() != bytes_per_tile)
                        throw std::runtime_error(
                            "read_tile_stack: tile is missing, out of grid, or failed to decode");
                    if (bytes_per_tile != 0)
                        std::memcpy(out->data() + i * bytes_per_tile,
                                    tile.data(), bytes_per_tile);
                    ++i;
                }
            } catch (...) {
                delete out;
                throw;
            }

            std::size_t shape[4] = {n, h, w, ch};
            nb::capsule owner(out, [](void* p) noexcept {
                delete static_cast<std::vector<std::uint8_t>*>(p);
            });
            return nb::ndarray<nb::numpy>(
                out->data(), 4, shape, owner, nullptr,
                nb::dlpack::dtype{(std::uint8_t)nb::dlpack::dtype_code::UInt, 8, 1});
        }, nb::arg("level"), nb::arg("tiles"),
           nb::arg("rgba") = true, nb::arg("fill_missing") = false);
    m.def("wsi_open", [](nb::list instances) {
        std::vector<std::vector<std::byte>> insts;
        insts.reserve(instances.size());
        for (auto item : instances) {
            nb::bytes b = nb::cast<nb::bytes>(item);
            const auto* p = reinterpret_cast<const std::byte*>(b.c_str());
            insts.emplace_back(p, p + b.size());
        }
        return dcmbase::wsi::slide(std::move(insts));
    }, nb::arg("instances"), "Open a WSI pyramid from a list of instance bytes → WsiSlide.");
    m.def("wsi_open_paths", [](nb::list paths) {
        std::vector<std::string> insts;
        insts.reserve(paths.size());
        for (auto item : paths) {
            insts.push_back(nb::cast<std::string>(item));
        }
        return dcmbase::wsi::slide(std::move(insts));
    }, nb::arg("paths"), "Open a file-backed WSI pyramid from a list of DICOM instance paths → WsiSlide.");
    m.def("decode", &decode, nb::arg("path"), nb::arg("frame") = 0, nb::arg("rescale") = false,
          "Decode a DICOM file to (ndarray[frames, rows, cols(, samples)], meta dict). "
          "rescale=True returns dcmcore's modality-LUT (HU) float.");
    m.def("decode_bytes", &decode_bytes, nb::arg("part10"), nb::arg("frame") = 0,
          nb::arg("rescale") = false,
          "Decode in-memory Part-10 bytes to (ndarray, meta) — same engine as decode(), "
          "for a from-scratch Dataset (no backing file).");
    m.def("n_frames", &n_frames, nb::call_guard<nb::gil_scoped_release>(), nb::arg("path"),
          "Number of decodable frames in a DICOM file.");
    m.def("assemble_4d", &assemble_4d, nb::arg("paths"),
          "Assemble a 4D stack (ndarray[V,depth,rows,cols] + meta{affine,spacing,volume_path,"
          "volume_frame,dimensions,...}); the non-Z axes (DWI direction / fMRI time / echo / "
          "cardiac phase) form the 4th dim, labelled by meta['dimensions'].");
    m.def("assemble_dwi", &assemble_dwi, nb::arg("paths"), nb::arg("order") = "gradient",
          "Group single-frame files by CSA diffusion and assemble a 4D DWI stack "
          "(ndarray[V,Z,Y,X] + meta{bvals, bvecs voxel-frame, affine, spacing}). "
          "order='gradient' (b0 first) or 'acquisition' (InstanceNumber order).");
    m.def("assemble_volume", &assemble_volume, nb::arg("paths"),
          "Assemble a spatially-ordered 3D HU volume from a series via dcmbase's "
          "vol_* engine. Returns (ndarray[depth, rows, cols] float32, meta{spacing, affine, ...}).");
    m.def("write_nifti_volume", &write_nifti_volume, nb::arg("array"), nb::arg("affine"), nb::arg("path"),
          "Write a [depth,rows,cols] float32 volume to NIfTI-1 (.nii/.nii.gz) via "
          "dcmbase's dcm_nifti_write. affine is 16 floats, column-major voxel→world LPS.");
    m.def("read_nifti", &read_nifti, nb::arg("path"),
          "Read a NIfTI-1 file (.nii/.nii.gz). Returns (ndarray[depth,rows,cols], "
          "meta{affine column-major LPS, spacing[z,y,x], scl_slope, scl_inter}).");

    // pydcm.transforms (dcmbase::transform CPU — the deterministic 'ITK' layer)
    m.def("transform_affine", &transform_affine, nb::arg("array"), nb::arg("affine"),
          nb::arg("matrix"), nb::arg("is_label") = false, nb::arg("interp") = "linear",
          "Apply a column-major voxel→voxel affine to the image (MONAI Affine); output keeps "
          "the source grid. Returns (arr, meta).");
    m.def("transform_resample_separate_z", &transform_resample_separate_z,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_shape"),
          "nnU-Net anisotropic separate-z resample to out_shape[D,H,W]: in-plane cubic "
          "B-spline + nearest through-plane, fp64 prefilter (scipy-faithful). Returns (arr, meta).");
    m.def("transform_resample_cubic", &transform_resample_cubic,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_shape"),
          "Isotropic cubic B-spline resample to out_shape[D,H,W] (scipy order=3, fp64, "
          "bit-exact with skimage.resize order=3). Returns (arr, meta).");
    m.def("transform_pil_resize2d", &transform_pil_resize2d,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_h"), nb::arg("out_w"),
          nb::arg("filter") = "bicubic",
          "Pillow-faithful 2-D resize per slice to (out_h,out_w) — filter 'bicubic'/'bilinear', "
          "anti-aliased downscale; bit-exact with PIL.Image.resize (uint8 fixed-point / float). "
          "Returns (arr, meta).");
    m.def("transform_bilinear_resize2d", &transform_bilinear_resize2d,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_h"), nb::arg("out_w"),
          "Plain (no anti-alias) bilinear 2-D resize per slice to (out_h,out_w) — half-pixel, "
          "clamp, double; the deployment / cv2 INTER_LINEAR / F.interpolate convention "
          "(bit-exact w/ ai_segmeation's resizeBilinearU8). Returns (arr, meta).");
    m.def("transform_resample_nearest", &transform_resample_nearest,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_shape"),
          "Nearest resample to out_shape[D,H,W] (scipy order=0, half-pixel) — for labels. "
          "Returns (arr, meta).");
    m.def("transform_resample_grid_sample", &transform_resample_grid_sample,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_shape"),
          "Trilinear resample to out_shape[D,H,W] matching torch/MONAI grid_sample "
          "(align_corners=False, bilinear, border), fp64. Returns (arr, meta).");
    m.def("transform_resample", &transform_resample,
          nb::arg("array"), nb::arg("affine"), nb::arg("out_shape"), nb::arg("backend"),
          "Resample to out_shape[D,H,W] under an interpolation convention "
          "('skimage'/'torch'/'itk'). Images only; use nearest for labels. Returns (arr, meta).");
    m.def("transform_resample_to_reference", &transform_resample_to_reference,
          nb::arg("moving"), nb::arg("moving_affine"), nb::arg("ref_shape"), nb::arg("ref_affine"),
          nb::arg("is_label") = false, nb::arg("interp") = "linear", nb::arg("fill") = 0.0,
          "Resample moving[D,H,W] onto a reference grid (ref_shape[D,H,W] + ref_affine). "
          "The inverse/round-trip primitive: map a prediction back onto a reference Volume's "
          "grid; out-of-bounds voxels get `fill` (sitk defaultPixelValue). Returns (arr, meta).");
    m.def("transform_resample_to_spacing", &transform_resample_to_spacing,
          nb::arg("array"), nb::arg("affine"), nb::arg("spacing"),
          nb::arg("is_label") = false, nb::arg("interp") = "linear",
          "Resample [D,H,W] to an axis-aligned LPS grid at spacing[x,y,z] mm. "
          "is_label=True forces nearest (no class blending). Returns (arr, meta{affine}).");
    m.def("transform_normalize_zscore", &transform_normalize_zscore,
          nb::arg("array"), nb::arg("affine"), nb::arg("nonzero") = false,
          "z-score normalize (→float32). nonzero=True ignores zero voxels. Returns (arr, meta).");
    m.def("transform_scale_intensity_range", &transform_scale_intensity_range,
          nb::arg("array"), nb::arg("affine"), nb::arg("a_min"), nb::arg("a_max"),
          nb::arg("b_min"), nb::arg("b_max"), nb::arg("clip") = true,
          "Linear remap [a_min,a_max]→[b_min,b_max] (→float32; CT windowing). Returns (arr, meta).");
    m.def("transform_normalize_ct", &transform_normalize_ct,
          nb::arg("array"), nb::arg("affine"), nb::arg("clip_lo"), nb::arg("clip_hi"),
          nb::arg("mean"), nb::arg("std"),
          "Clip to [clip_lo,clip_hi] then z-score with fixed mean/std (nnU-Net "
          "CTNormalization / MONAI NormalizeIntensity+clip; →float32). Returns (arr, meta).");
    m.def("transform_rescale_robust", &transform_rescale_robust,
          nb::arg("array"), nb::arg("affine"), nb::arg("dst_min"), nb::arg("dst_max"),
          nb::arg("f_low"), nb::arg("f_high"),
          "FreeSurfer/FastSurfer conform robust histogram rescale to [dst_min,dst_max] "
          "(f_low/f_high robust crops; →float32). Returns (arr, meta).");
    m.def("transform_scale_intensity_range_percentiles", &transform_scale_intensity_range_percentiles,
          nb::arg("array"), nb::arg("affine"), nb::arg("lower"), nb::arg("upper"),
          nb::arg("b_min"), nb::arg("b_max"), nb::arg("clip") = true,
          "Like scale_intensity_range but a_min/a_max are per-image percentiles lower/upper "
          "(0..100, np.percentile linear; MONAI ScaleIntensityRangePercentiles). Returns (arr, meta).");
    m.def("transform_adjust_contrast", &transform_adjust_contrast,
          nb::arg("array"), nb::arg("affine"), nb::arg("gamma"),
          "Gamma contrast ((x-min)/(range+1e-7))^gamma*range+min (MONAI AdjustContrast; "
          "→float32). Returns (arr, meta).");
    m.def("transform_argmax", &transform_argmax, nb::arg("probs"), nb::arg("affine"),
          "Channel-last argmax: probs[D,H,W,C] → label[D,H,W] (uint8 if C≤256 else uint16). "
          "Returns (labels, meta{affine}).");
    m.def("transform_gaussian_smooth", &transform_gaussian_smooth,
          nb::arg("array"), nb::arg("affine"), nb::arg("sigma"),
          "Separable Gaussian smoothing, per-axis sigma[z,y,x] in voxels (→float32; "
          "MONAI GaussianSmooth). Returns (arr, meta).");
    m.def("transform_resize", &transform_resize, nb::arg("array"), nb::arg("affine"),
          nb::arg("out_shape"), nb::arg("is_label") = false, nb::arg("interp") = "linear",
          "Resample [D,H,W] to exactly out_shape[D,H,W] voxels over the same FOV. Returns (arr, meta).");
    m.def("transform_crop", &transform_crop, nb::arg("array"), nb::arg("affine"),
          nb::arg("start"), nb::arg("size"),
          "Crop the [z,y,x] box [start, start+size). Exact. Returns (arr, meta{affine shifted}).");
    m.def("transform_pad", &transform_pad, nb::arg("array"), nb::arg("affine"),
          nb::arg("lo"), nb::arg("hi"), nb::arg("mode") = "constant", nb::arg("value") = 0.0f,
          "Pad lo/hi [z,y,x] voxels. mode='constant'|'edge'|'reflect'. Returns (arr, meta).");
    m.def("transform_crop_foreground", &transform_crop_foreground, nb::arg("array"),
          nb::arg("affine"), nb::arg("margin") = 0u,
          "Crop to the non-zero bounding box + margin. Returns (arr, meta).");
    m.def("transform_flip", &transform_flip, nb::arg("array"), nb::arg("affine"), nb::arg("axis"),
          "Reverse voxel order along flagged [z,y,x] axes. Exact. Returns (arr, meta).");
    m.def("transform_transpose", &transform_transpose, nb::arg("array"), nb::arg("affine"), nb::arg("axes"),
          "Permute voxel axes by a (z,y,x)-order permutation of (0,1,2). Exact. Returns (arr, meta).");
    m.def("transform_center_crop", &transform_center_crop, nb::arg("array"), nb::arg("affine"),
          nb::arg("size"), nb::arg("is_label") = false,
          "Center-crop to size[z,y,x] (MONAI CenterSpatialCrop; clamps to source, no pad). "
          "Returns (arr, meta{affine shifted}).");
    m.def("transform_spatial_pad", &transform_spatial_pad, nb::arg("array"), nb::arg("affine"),
          nb::arg("size"), nb::arg("mode") = "constant", nb::arg("value") = 0.0f, nb::arg("is_label") = false,
          "Centered pad to at least size[z,y,x] (MONAI SpatialPad symmetric). Returns (arr, meta).");
    m.def("transform_divisible_pad", &transform_divisible_pad, nb::arg("array"), nb::arg("affine"),
          nb::arg("k"), nb::arg("mode") = "constant", nb::arg("value") = 0.0f, nb::arg("is_label") = false,
          "Centered pad so each [z,y,x] axis is a multiple of k[axis] (MONAI DivisiblePad). "
          "Returns (arr, meta).");
    m.def("transform_rotate90", &transform_rotate90, nb::arg("array"), nb::arg("affine"),
          nb::arg("k"), nb::arg("axis0") = 1, nb::arg("axis1") = 2, nb::arg("is_label") = false,
          "Rotate k*90° in the (axis0,axis1) plane — axes are numpy [D,H,W] indices (MONAI "
          "Rotate90 / np.rot90). Exact; affine updates. Returns (arr, meta).");
    m.def("transform_sliding_window_positions", &transform_sliding_window_positions,
          nb::arg("spatial"), nb::arg("roi"), nb::arg("overlap") = 0.25,
          "Sliding-window patch origins over spatial[z,y,x] with window roi[z,y,x] and "
          "overlap (MONAI dense_patch_slices). Flat [z,y,x] origins, 3 per patch.");
    m.def("transform_gaussian_importance_map", &transform_gaussian_importance_map,
          nb::arg("roi"), nb::arg("sigma_scale") = 0.125, nb::arg("convention") = "nnunet",
          "Gaussian blend-weight window roi[z,y,x], sigma=roi*sigma_scale. convention="
          "'nnunet' (center roi//2, peak 1) or 'monai' (center (roi-1)/2, min-clamped 1e-3). "
          "Returns (map[D,H,W] float32, meta).");
    m.def("transform_reorient", &transform_reorient, nb::arg("array"), nb::arg("affine"), nb::arg("axcodes"),
          "Reorient to target axcodes (3 of L/R/P/A/S/I; world is LPS). Exact axis "
          "permutation + flips. Returns (arr, meta{affine}).");
    m.def("transform_connected_components", &transform_connected_components,
          nb::arg("array"), nb::arg("affine"), nb::arg("connectivity") = 6,
          "Label connected components of the non-zero foreground (1..N) → uint16. "
          "connectivity 6/18/26. Returns (labels, meta).");
    m.def("transform_keep_largest_cc", &transform_keep_largest_cc,
          nb::arg("array"), nb::arg("affine"), nb::arg("connectivity") = 6, nb::arg("per_class") = true,
          "Keep the largest connected component, zero the rest (MONAI "
          "KeepLargestConnectedComponent). per_class=True handles each class independently. "
          "Returns (labels, meta).");
    m.def("transform_fill_holes", &transform_fill_holes,
          nb::arg("array"), nb::arg("affine"), nb::arg("connectivity") = 6,
          "Fill enclosed background holes per class (MONAI FillHoles). connectivity "
          "6/18/26. Returns (labels, meta).");
    m.def("transform_as_discrete", &transform_as_discrete,
          nb::arg("array"), nb::arg("affine"), nb::arg("threshold"),
          "Binarize at threshold (value>threshold → 1) → uint8 (MONAI AsDiscrete). Returns (arr, meta).");
    m.def("transform_remove_small_objects", &transform_remove_small_objects,
          nb::arg("array"), nb::arg("affine"), nb::arg("min_size"),
          nb::arg("connectivity") = 6, nb::arg("per_class") = true,
          "Zero connected components smaller than min_size (MONAI RemoveSmallObjects). "
          "Returns (labels, meta).");
    m.def("build_seg_from_nifti", &build_seg_from_nifti,
          nb::arg("ref_paths"), nb::arg("mask_nii"), nb::arg("segments"),
          "NIfTI label volume + reference series -> coded DICOM-SEG Part-10 bytes "
          "(affine-aware Z alignment; segments = list of dicts).");
    m.def("mosaic_expand", &mosaic_expand, nb::arg("part10_bytes"),
          "Expand a Siemens mosaic (CSA-driven) into (ndarray[N,rows,cols], meta{ipp,iop,"
          "slice_normal,spacing}); returns None if the bytes are not a mosaic.");
    m.def("siemens_diffusion", &siemens_diffusion, nb::arg("part10_bytes"),
          "Siemens DWI b-value + gradient direction from the CSA header: "
          "{b_value, gradient:[x,y,z]} or None (the .bval/.bvec source).");
    m.def("read_diffusion", &read_diffusion, nb::arg("part10_bytes"),
          "Unified per-frame DWI: [{b_value, gradient:[x,y,z]}, ...] from the standard "
          "MR Diffusion sequence (enhanced-MF) or legacy Siemens CSA; None if neither.");
    m.def("bids_sidecar", &bids_sidecar, nb::arg("part10_bytes"),
          "BIDS JSON sidecar string (timing in seconds, sequence, geometry) extracted from one DICOM instance's bytes.");
    m.def("dti_fit_maps", &dti_fit_maps, nb::arg("b0"), nb::arg("dwi"), nb::arg("bvals"),
          nb::arg("bvecs"), nb::arg("maps"), nb::arg("wls") = false,
          "Native DTI: OLS (or WLS) tensor fit -> eigen -> scalar maps. Returns {name: ndarray}. "
          "FA/MD/AD/RD float; CL/CP/CS = Westin-1997 (/lambda1); linearity/planarity/sphericity "
          "= trace-normalized (dipy); DEC uint8 [n_voxels,4].");
    m.def("dti_eigen", &dti_eigen, nb::arg("b0"), nb::arg("dwi"), nb::arg("bvals"),
          nb::arg("bvecs"), nb::arg("wls") = false,
          "Native DTI tensor fit -> (eigenvalues[n_voxels,3], eigenvectors[n_voxels,9]).");
    m.def("dti_track", &dti_track_streamlines,
          nb::arg("evecs"), nb::arg("fa"), nb::arg("cols"), nb::arg("rows"), nb::arg("depth"),
          nb::arg("spacing"), nb::arg("fa_threshold") = 0.15f, nb::arg("angle_threshold") = 0.7071f,
          nb::arg("step_size") = 0.5f, nb::arg("max_steps") = 2000u, nb::arg("seed_fa_min") = 0.3f,
          nb::arg("max_tracks") = 100000u, nb::arg("max_total_points") = 10000000u,
          "Native dcm_dti deterministic RK4 tractography -> list of [P,3] streamlines in mm "
          "(voxel*spacing; the streamlines dcmrender/dcm_fiber renders for medfilm).");
    m.def("scan_dicom_dir", &scan_dicom_dir, nb::arg("root"), nb::arg("recursive") = true,
          "Discover DICOM files under a directory (extension or DICM-preamble detection).");
    m.def("read_json", &read_json, nb::call_guard<nb::gil_scoped_release>(), nb::arg("path"), nb::arg("charset_override") = std::string{},
          nb::arg("inline_binary") = true,
          "Read a DICOM file to a DICOM JSON Model string (all elements, charset-decoded; "
          "non-pixel binary inlined as base64 unless inline_binary=False).");
    m.def("encode_ivr", &encode_ivr, nb::arg("json"),
          "DICOM JSON Model dataset -> bare Implicit VR LE bytes (no meta / SOP-UID "
          "requirement). Reuses bridge::json_to_ivr; backs DIMSE query/identifier encoding.");
    m.def("write_part10", &write_part10, nb::arg("json"), nb::arg("transfer_syntax") = "",
          "Build Part-10 file bytes from a DICOM JSON Model object string. transfer_syntax="
          "'1.2.840.10008.1.2.1' emits Explicit VR LE (recoded from the IVR-LE build); "
          "empty / IVR-LE keeps Implicit VR LE.");
    m.def("transcode", &transcode, nb::arg("part10"), nb::arg("target_ts"),
          "Re-encode a Part-10 buffer to target_ts (encapsulated lossless: JPEG2000 .90 / "
          "JPEG-LS .80 / HTJ2K .201 / JPEG-XL). Backs pydcm Dataset.compress.");
    m.def("read_file_meta", &read_file_meta, nb::arg("path"),
          "Part-10 file-meta (group 0002): {has_meta, transfer_syntax, sop_class, sop_instance}. "
          "Naked datasets get a sniffed transfer_syntax and has_meta=False.");
    m.def("read_meta_json", &read_meta_json, nb::call_guard<nb::gil_scoped_release>(), nb::arg("path"),
          "Full group-0002 File Meta Information as a DICOM JSON Model string (every "
          "element, not just the 3 UIDs). Empty for a naked dataset. Reuses dataset_to_json.");
    m.def("read_pixel_data", &read_pixel_data, nb::arg("path"),
          "Raw (7FE0,0010) PixelData value bytes, or None when absent / a transfer syntax "
          "this fast path skips (deflate/EVR-BE). Reuses dataset::parse; backs ds.PixelData.");
    m.def("has_pixel_data", &has_pixel_data, nb::call_guard<nb::gil_scoped_release>(), nb::arg("path"),
          "True if the dataset contains (7FE0,0010) PixelData, without copying the bytes — "
          "backs the lazy presence of PixelData in the Dataset mapping protocol.");
    m.def("pixel_data_vr", &pixel_data_vr, nb::arg("path"),
          "On-disk VR of (7FE0,0010) for Explicit-VR files ('OB'/'OW'/'OF'/'OD'), else None — "
          "lets the lazy ds.PixelData keep the file's real VR instead of guessing from Bits Allocated.");
    m.def("mint_uid", &mint_uid, nb::arg("seed") = std::string{}, nb::arg("root") = std::string{},
          "Mint a DICOM UID via dcmbase::uid::mint (the canonical generator). Deterministic "
          "per seed; backs pydcm.uid.generate_uid.");
    m.def("edit_part10", &edit_part10, nb::arg("original"), nb::arg("ops"),
          "Apply (tag,kind,value,vr) edit ops to an original Part-10 buffer byte-verbatim "
          "(keeps Transfer Syntax + PixelData). kind is 'modify'/'insert'/'erase'.");
    m.def("deidentify", &deidentify, nb::arg("data"), nb::arg("options"),
          "De-identify one Part-10 buffer (PS3.15 Annex E) via dcmbase::deident::session; "
          "returns the de-identified bytes. `options` is a dict (see pydcm.deident).");
    m.def("deidentify_series", &deidentify_series, nb::arg("files"), nb::arg("options"),
          "De-identify a list of Part-10 buffers through ONE session — UID remap stays "
          "consistent across the batch (a study's cross-references survive).");
    m.def("clean_pixel_data", &clean_pixel_data, nb::arg("data"),
          nb::arg("regions") = nb::none(), nb::arg("use_ctp") = true,
          nb::arg("require_match") = false,
          "Black out burned-in PHI regions via dcmbase::pixanon (113101 Clean Pixel Data). "
          "regions = list of (x,y,w,h[,frame]) pixel rects; use_ctp matches the RSNA CTP "
          "device-signature library. Re-emits uncompressed (never re-compresses).");
    m.def("build_dicomdir", &build_dicomdir, nb::arg("inputs"),
          nb::arg("file_set_id") = std::string{},
          "Build a conformant (Explicit-VR-LE) DICOMDIR over dcmbase::dicomdir::build. "
          "inputs = list of (part10_bytes, media_relative_file_id); returns DICOMDIR bytes.");
    m.def("tag_for_keyword", &tag_for_keyword, nb::arg("keyword"),
          "DICOM keyword (e.g. 'PatientName') -> packed tag (group<<16|element), or None.");
    m.def("describe_tag", &describe_tag, nb::arg("tag"),
          "Packed standard tag -> {keyword,name,vr,vm,retired} from the native union dict, or None.");
    m.def("describe_private", &describe_private, nb::arg("creator"), nb::arg("group"),
          nb::arg("elem_low"),
          "Private (creator, group, element-low-byte) -> {keyword,name,vr,vm,retired}, or None.");
    m.def("uid_lookup", &uid_lookup, nb::arg("uid"),
          "UID -> {name,type,keyword,info,retired,cid} from the native UID dict, or None.");
    m.def("uid_for_keyword", &keyword_to_uid, nb::arg("keyword"),
          "Public UID keyword (e.g. 'CTImageStorage') -> UID string, or '' if unknown.");
    m.def("uid_table", &uid_table,
          "The full native UID dictionary as (uid,name,type,keyword,info,retired,cid) tuples.");
    m.def("content_json", &content_json, nb::arg("path"),
          nb::arg("contours") = false, nb::arg("control_points") = false,
          "Semantic JSON (str) of a structured object (SEG/RTSTRUCT/RTPLAN/RTDOSE/PS/"
          "Waveform/Ophthalmic Visual Field) via dcmbase::content::to_json; None if not "
          "a structured object. Backs pydcm.content.");
    m.def("sr_to_html", &sr_to_html, nb::arg("path"),
          "SR document -> clinical-readable HTML (str) via dcmbase::sr::to_html (dcm_sr_html "
          "C core) — the same renderer behind the CLI dsr2html, byte-identical; handles any SR. "
          "Backs pydcm.sr_to_html.");
    m.def("iod_validate", &iod_validate, nb::arg("path"),
          "IOD/module Type-1/2 conformance (dciodvfy core) for the file's SOP Class -> "
          "list of {severity,tag,module,message}. Reuses dcmbase::iod::validate. Backs "
          "pydcm.iod_validate / OPVDicom.check_dicom_compliance.");
    m.def("read_rtdose", &read_rtdose, nb::arg("path"),
          "RT Dose file -> (ndarray[depth,rows,cols] float32 scaled dose, meta dict). "
          "Scaling/geometry/DVH decode run in dcmbase::rt (C++). Backs pydcm.rt.");
    m.def("encapsulate", &encapsulate_py,
          nb::arg("payload"), nb::arg("type"), nb::arg("title") = "",
          nb::arg("mime") = "", nb::arg("units") = "", nb::arg("ids") = nb::dict(),
          "Wrap a document payload into its Encapsulated Document Part-10 (bytes) "
          "via dcmbase::encap (the dcmencap engine). Backs pydcm.write_encapsulated.");
    m.def("read_encapsulated", &read_encapsulated_py, nb::arg("path"),
          "Extract an Encapsulated Document: dict(payload bytes, mime, title, "
          "sop_class_uid, sop_instance_uid, type). Backs pydcm.read_encapsulated.");
    m.def("encap_detect", &encap_detect_py, nb::arg("filename"), nb::arg("head"),
          "Detect a document type (pdf|cda|stl|obj|mtl) from filename + content "
          "magic via dcmbase::encap::detect; None when unknown.");
    m.def("compute_dvh", &compute_dvh_py,
          nb::arg("rtstruct_path"), nb::arg("rtdose_path"), nb::arg("roi"),
          nb::arg("limit") = 0, nb::arg("calculate_full_volume") = true,
          nb::arg("thickness") = 0.0,
          "DVH of one ROI from RTSTRUCT+RTDOSE (validated DVH parity, "
          "computed in dcmbase::rt). Returns dict with differential/cumulative "
          "counts (cm^3 per cGy bin), volume, min/max/mean (Gy). Backs pydcm.rt.");
    m.def("write_rtdose", &write_rtdose_py,
          nb::arg("dose"), nb::arg("origin"), nb::arg("orientation"),
          nb::arg("ps_row"), nb::arg("ps_col"), nb::arg("offsets"),
          nb::arg("units") = "GY", nb::arg("dose_type") = "PHYSICAL",
          nb::arg("summation") = "PLAN", nb::arg("ref_plan_uid") = "",
          nb::arg("patient_name") = "", nb::arg("patient_id") = "",
          nb::arg("study_uid") = "", nb::arg("study_date") = "",
          nb::arg("series_uid") = "", nb::arg("frame_of_ref_uid") = "",
          nb::arg("scaling") = 0.0, nb::arg("bits") = 32,
          "Author an RT Dose Part-10 (bytes) from a float64 [D,R,C] grid via "
          "dcmbase's dcm_rtdose_export (quantisation in C). Backs pydcm.rt.write_rtdose.");
    m.def("radiomics_features", &radiomics_features,
          nb::arg("pixels"), nb::arg("mask"),
          nb::arg("spacing_x") = 1.0f, nb::arg("spacing_y") = 1.0f, nb::arg("spacing_z") = 1.0f,
          nb::arg("bins") = 32, nb::arg("range_min") = -1024.0f, nb::arg("range_max") = 3071.0f,
          nb::arg("bin_width") = 0.0f, nb::arg("resample_spacing") = 0.0f,
          nb::arg("normalize") = false, nb::arg("normalize_scale") = 1.0f,
          nb::arg("log_sigmas") = nb::list(), nb::arg("wavelet") = false, nb::arg("averaged") = true,
          nb::arg("resegment") = false, nb::arg("resegment_sigma") = false,
          nb::arg("reseg_min") = 0.0f, nb::arg("reseg_max") = 0.0f, nb::arg("resample_bspline") = false,
          nb::arg("voxel_array_shift") = 0.0f, nb::arg("filters") = nb::list(),
          nb::arg("distances") = nb::list(),
          "IBSI radiomic features over an ROI (float32 pixels + uint8 mask, 2D/3D) via "
          "dcmbase::radiomics::extract -> {name: value}. With log_sigmas / wavelet, runs the "
          "filtered passes too (filter-prefixed keys). Backs pydcm.radiomics (array form).");
    m.def("radiomics_features_prepared", &radiomics_features_prepared,
          nb::arg("pixels"), nb::arg("mask"),
          nb::arg("spacing_x") = 1.0f, nb::arg("spacing_y") = 1.0f, nb::arg("spacing_z") = 1.0f,
          nb::arg("bins") = 32, nb::arg("range_min") = -1024.0f, nb::arg("range_max") = 3071.0f,
          nb::arg("bin_width") = 0.0f, nb::arg("resample_spacing") = 0.0f,
          nb::arg("normalize") = false, nb::arg("normalize_scale") = 1.0f, nb::arg("averaged") = true,
          nb::arg("resegment") = false, nb::arg("resegment_sigma") = false,
          nb::arg("reseg_min") = 0.0f, nb::arg("reseg_max") = 0.0f, nb::arg("resample_bspline") = false,
          nb::arg("voxel_array_shift") = 0.0f, nb::arg("distances") = nb::list(),
          "Standard IBSI features (original image) PLUS the preprocessed/discretised grid "
          "(image/mask/levels as (nz,h,w) arrays + nb/spacing/range) for host-side custom "
          "features over the SAME grid — dcmbase::radiomics::extract_prepared. Returns "
          "(features, roi). Backs pydcm.radiomics's custom-feature hook (array form).");
    m.def("radiomics_file", &radiomics_file,
          nb::arg("image"), nb::arg("mask"), nb::arg("roi_min"), nb::arg("roi_max"),
          nb::arg("bins") = 32, nb::arg("range_min") = -1024.0f, nb::arg("range_max") = 3071.0f,
          nb::arg("bin_width") = 0.0f, nb::arg("resample_spacing") = 0.0f,
          nb::arg("normalize") = false, nb::arg("normalize_scale") = 1.0f,
          nb::arg("log_sigmas") = nb::list(), nb::arg("wavelet") = false, nb::arg("averaged") = true,
          nb::arg("resegment") = false, nb::arg("resegment_sigma") = false,
          nb::arg("reseg_min") = 0.0f, nb::arg("reseg_max") = 0.0f, nb::arg("resample_bspline") = false,
          nb::arg("voxel_array_shift") = 0.0f, nb::arg("filters") = nb::list(),
          nb::arg("distances") = nb::list(),
          "IBSI features straight from an image DICOM (+ a co-framed mask DICOM, else the "
          "[roi_min,roi_max] threshold) via dcmbase::radiomics::extract_file -- the same "
          "decode/spacing/mask orchestration as the dcmradiomics CLI. Backs pydcm.radiomics "
          "(file form).");
    // ── DCE-MRI pharmacokinetic modelling (pydcm.dce) ──
    m.def("dce_parker_aif", &dce_parker_aif, nb::arg("times_min"), nb::arg("hct") = 0.0,
          "Parker (2006) population arterial input function sampled on times_min (min). "
          "hct=0 returns the published plasma curve verbatim. -> AIF[T] (mM).");
    m.def("dce_population_aif", &dce_population_aif, nb::arg("model"), nb::arg("times_min"),
          nb::arg("hct") = 0.0,
          "Population AIF by name: parker | georgiou | fritz_hansen | weinmann | mcgrath, "
          "sampled on times_min (min). hct=0 = plasma verbatim. -> AIF[T] (mM). "
          "Backs pydcm.dce.population_aif.");
    m.def("dce_forward", &dce_forward_py, nb::arg("times_min"), nb::arg("cp"),
          nb::arg("model") = "ext_tofts", nb::arg("ktrans") = 0.1, nb::arg("ve") = 0.3,
          nb::arg("vp") = 0.02,
          "Synthesise a tissue concentration curve from PK params (Tofts/ext_tofts/patlak). "
          "-> Ct[T] (mM). Exact piecewise-linear-AIF convolution (dcmbase C core).");
    m.def("dce_signal_to_conc", &dce_signal_to_conc_py, nb::arg("signal"),
          nb::arg("n_baseline"), nb::arg("t1_0_s"), nb::arg("tr_s"), nb::arg("fa_deg"),
          nb::arg("r1"),
          "Invert the spoiled-GRE signal model -> tracer concentration[T] (mM).");
    m.def("dce_fit_curve", &dce_fit_curve, nb::arg("times_min"), nb::arg("ct"),
          nb::arg("cp"), nb::arg("model") = "ext_tofts",
          nb::arg("fit_delay") = false, nb::arg("delay_lo") = 0.0, nb::arg("delay_hi") = 0.5,
          "Fit one tissue curve -> {ktrans, ve, vp, rmse, iters, ok, delay}. Patlak=closed "
          "form, Tofts/ext_tofts=Levenberg-Marquardt. fit_delay adds a joint arterial-delay "
          "(min) search over [delay_lo, delay_hi] (dcmbase C core).");
    m.def("dce_fit_map", &dce_fit_map, nb::arg("series"), nb::arg("times_min"),
          nb::arg("model") = "ext_tofts", nb::arg("input") = "concentration",
          nb::arg("hct") = 0.0, nb::arg("measured_cp") = nb::none(),
          nb::arg("mask") = nb::none(), nb::arg("t1_0_s") = 1.4, nb::arg("tr_s") = 0.005,
          nb::arg("fa_deg") = 25.0, nb::arg("r1") = 4.5, nb::arg("n_baseline") = 0,
          nb::arg("enhance_thresh") = 0.0f, nb::arg("t1_map") = nb::none(),
          nb::arg("fit_delay") = false, nb::arg("delay_lo") = 0.0, nb::arg("delay_hi") = 0.5,
          "Voxel-wise PK fit over a [T,H,W] float32 series via dcmbase::dce::fit_slice "
          "(shared with the CLI/server). input='spgr' converts signal->conc per voxel first. "
          "fit_delay adds a per-voxel joint arterial-delay (min) search. "
          "-> {ktrans[H,W], ve, vp, rmse, fitted, [delay]}. Backs pydcm.dce.fit.");
    m.def("dce_t1_map_vfa", &dce_t1_map_vfa, nb::arg("volumes"), nb::arg("fa_deg"),
          nb::arg("tr_s"), nb::arg("mask") = nb::none(),
          "VFA/DESPOT1 baseline-T1 map from a [F,H,W] multi-flip-angle SPGR stack via "
          "dcmbase::dce::t1_map_vfa. -> {t1[H,W] (s), m0, fitted}. Backs pydcm.dce.t1_map_vfa.");

    m.def("write_seg", &write_seg,
          nb::arg("reference_paths"), nb::arg("labelmap"), nb::arg("segments"),
          nb::arg("output") = std::string{},
          "Author a coded BINARY DICOM Segmentation via dcmbase::seg::build (mkseg writer). "
          "Returns Part-10 bytes, or None when `output` is written. Backs pydcm.write_seg.");
    m.def("write_seg_fractional", &write_seg_fractional,
          nb::arg("reference_paths"), nb::arg("maps"), nb::arg("segments"),
          nb::arg("fractional_type") = 0, nb::arg("max_value") = 255,
          nb::arg("output") = std::string{},
          "Author a FRACTIONAL DICOM Segmentation (8-bit probability/occupancy maps, "
          "uint8 [nseg,(slices,)H,W]) via dcmbase::seg::build_fractional. Backs "
          "pydcm.write_seg(..., fractional=...).");
    m.def("write_sr", &write_sr, nb::arg("document"), nb::arg("output") = std::string{},
          "Author a Comprehensive Structured Report from a content-tree dict (same shape "
          "as the mksr JSON) via dcmbase::sr::build. Returns Part-10 bytes or None. Backs "
          "pydcm.write_sr.");
    m.def("write_report", &write_report, nb::arg("document"), nb::arg("output") = std::string{},
          "Author a TID 1500 Measurement Report SR from a measurements list via dcmbase's "
          "dcm_sr_export engine (the mkreport writer). Returns Part-10 bytes or None. Backs "
          "pydcm.write_report.");
    m.def("read_report", &read_report, nb::arg("path"),
          "Extract a TID 1500 SR's measurements -> {patient/study + measurements:[...]}, "
          "round-tripping write_report (empty list when not an SR). Backs pydcm.read_report.");
    m.def("write_measurement_report", &write_measurement_report, nb::arg("document"),
          nb::arg("output") = std::string{},
          "Author a TYPED TID 1500 Measurement Report (observer context + measurement groups "
          "with tracking / finding / finding sites / ROI / measurements + method/derivation / "
          "qualitative evaluations) via dcmbase::sr::build_measurement_report. Backs "
          "pydcm.write_measurement_report.");
    m.def("read_measurement_report", &read_measurement_report_py, nb::arg("path"),
          "Typed TID 1500 parse -> {patient/study, observer, groups:[...]}, round-tripping "
          "write_measurement_report. Backs pydcm.read_measurement_report.");
    m.def("write_ko", &write_ko, nb::arg("document"), nb::arg("output") = std::string{},
          "Author a Key Object Selection document (PS3.16 TID 2010) via dcmbase::ko::build. "
          "Returns Part-10 bytes or None. Backs pydcm.write_ko.");
    m.def("read_ko", &read_ko, nb::arg("path"),
          "Read a KOS -> {patient/study, title, references:[...]} (None when not a KOS). "
          "Backs pydcm.read_ko.");
    m.def("write_pr", &write_pr, nb::arg("document"), nb::arg("output") = std::string{},
          "Author a Grayscale Softcopy Presentation State (GSPS) via dcmbase's dcm_ps_export "
          "engine. Returns Part-10 bytes or None. Backs pydcm.write_pr.");
    m.def("write_paramap", &write_paramap,
          nb::arg("reference_paths"), nb::arg("values"), nb::arg("rwvm"),
          nb::arg("store_bits") = 0, nb::arg("store_signed") = 0,
          nb::arg("output") = std::string{},
          "Author a DICOM Parametric Map (real-valued array over the source series' "
          "geometry + Real World Value Mapping) via dcmbase's dcm_paramap_export engine "
          "(itkimage2paramap). store_bits 0 = float (FloatingPointImagePixel); 8/16 = "
          "integer pixels quantized through the RWVM. Returns Part-10 bytes or None. "
          "Backs pydcm.write_paramap.");
    m.def("write_legacy_converted", &write_legacy_converted,
          nb::arg("reference_paths"), nb::arg("options") = nb::dict{},
          nb::arg("output") = std::string{},
          "Fold a classic single-frame CT/MR/PET series into ONE Legacy Converted "
          "Enhanced multi-frame object via dcmbase::legacy::convert (the "
          "`legacy` capability). Inherits identity from the source, maps geometry/"
          "rescale/window/frame-type into Shared/Per-Frame Functional Groups, links "
          "each frame to its origin, and preserves leftover attributes verbatim. "
          "Returns Part-10 bytes or None. Backs pydcm.write_legacy_converted.");
    m.def("read_ann", &read_ann, nb::arg("path"),
          "Read a Microscopy Bulk Simple Annotations object -> {coordinate_type, groups:[...]} "
          "or None. Each group carries its coded property type/category, graphic type, and bulk "
          "point coordinates (raw float64 bytes) + the point index list. Backs pydcm.read_ann.");
    m.def("write_ann", &write_ann, nb::arg("document"), nb::arg("output") = std::string{},
          "Author a Microscopy Bulk Simple Annotations object via dcmbase::ann::build "
          "(inverse of read_ann). Returns Part-10 bytes or None. Backs pydcm.write_ann.");
    m.def("paramap_meta", &paramap_meta, nb::arg("path"),
          "Parametric-Map metadata -> {is_parametric_map, pixel_data_vr, is_float, rwvm}. "
          "Surfaces the Real World Value Mapping. Backs pydcm.read_paramap.");
    m.def("read_seg", &read_seg, nb::arg("path"), nb::arg("masks") = false,
          "Reconstruct a DICOM Segmentation -> (labelmap [slices,rows,cols] uint16 of segment "
          "numbers, meta) or, with masks=True, (per-segment occupancy [nseg,slices,rows,cols] "
          "float32 in [0,1], meta). None when not a segmentation. Backs pydcm.read_seg "
          "(segimage2itkimage) over dcmbase's dcm_seg_decode engine.");
    m.def("sr_code_meaning", &sr_code_meaning, nb::arg("scheme"), nb::arg("value"),
          "Code Meaning for (scheme, value) from the PS3.16 Content Mapping Resource "
          "(the most complete public set), or None if unknown. Backs pydcm.sr_code_meaning.");
    m.def("sr_validate_code", &sr_validate_code, nb::arg("scheme"), nb::arg("value"),
          nb::arg("meaning") = std::string{},
          "True if (scheme, value) is a known coded concept and (if given) its meaning "
          "matches. Backs pydcm.sr_validate_code.");
    m.def("sr_cid_has", &sr_cid_has, nb::arg("cid"), nb::arg("scheme"), nb::arg("value"),
          "True if the coded concept is a member of Context Group `cid`. Backs pydcm.sr_cid_has.");
    m.def("sr_validate", &sr_validate, nb::arg("path"),
          "Validate an SR's content tree (structural + coded-concept + TID content-template "
          "conformance) -> a list of {severity, location, message} dicts. Backs pydcm.sr_validate.");
}
