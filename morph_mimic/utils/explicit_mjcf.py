"""Materialize MJCF defaults for importers with incomplete class support.

The source assets remain untouched. Derived files are accepted only while
their source/mesh hashes match, and are recompiled in MuJoCo before publishing.
This establishes MuJoCo equivalence, not PhysX importer equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_explicit_mjcf(source_path, output_dir):
    """Read a previously verified derivative without needing MuJoCo installed."""
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    output = output_dir / f"{source_path.stem}_explicit.xml"
    provenance = output.with_suffix(".json")
    guidance = "Run python -m morph_mimic.scripts.prepare_runtime_mjcf --robot g1 in the MuJoCo Python environment."
    if not output.is_file() or not provenance.is_file():
        raise ValueError(f"Explicit reference MJCF cache is missing. {guidance}")
    data = json.loads(provenance.read_text())
    if data.get("schema") != 1 or data.get("source_path") != str(source_path):
        raise ValueError(f"Explicit MJCF provenance does not match its source. {guidance}")
    if sha256(output) != data.get("derived_sha256"):
        raise ValueError(f"Explicit MJCF cache hash mismatch. {guidance}")
    for filename, expected in data.get("dependency_sha256", {}).items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Explicit MJCF source dependency changed: {path}. {guidance}")
    if data.get("dependency_sha256", {}).get(str(source_path)) != sha256(source_path):
        raise ValueError(f"Explicit MJCF source is not verified. {guidance}")
    if not data.get("mujoco_equivalence_verified"):
        raise ValueError(f"Explicit MJCF has no equivalence verification. {guidance}")
    return output


def _numbers(values):
    return " ".join(format(float(value), ".17g") for value in np.asarray(values).reshape(-1))


def _equivalence(source, derived):
    import mujoco

    for name in ("nq", "nv", "nbody", "njnt", "ngeom", "nmesh", "nexclude"):
        if getattr(source, name) != getattr(derived, name):
            raise ValueError(f"explicit MJCF changes {name}")
    checked = [
        "body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
        "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range", "jnt_stiffness",
        "qpos0", "dof_armature", "dof_damping", "dof_frictionloss", "geom_bodyid", "geom_type",
        "geom_dataid", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
        "geom_condim", "geom_friction", "geom_margin", "geom_gap", "geom_solref", "geom_solimp",
        "exclude_signature",
    ]
    for name in checked:
        if not np.allclose(getattr(source, name), getattr(derived, name), rtol=1e-10, atol=1e-10):
            raise ValueError(f"explicit MJCF changes model.{name}")
    for obj, count in ((mujoco.mjtObj.mjOBJ_BODY, source.nbody),
                       (mujoco.mjtObj.mjOBJ_JOINT, source.njnt)):
        for index in range(count):
            if mujoco.mj_id2name(source, obj, index) != mujoco.mj_id2name(derived, obj, index):
                raise ValueError("explicit MJCF changes body/joint names")
    rng = np.random.default_rng(143)
    for _ in range(4):
        q = source.qpos0.copy()
        q[:3] = rng.normal(size=3)
        quaternion = rng.normal(size=4)
        q[3:7] = quaternion / np.linalg.norm(quaternion)
        for index in range(source.njnt):
            if source.jnt_type[index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = source.jnt_range[index] if source.jnt_limited[index] else (-1, 1)
                q[source.jnt_qposadr[index]] = rng.uniform(lo, hi)
        first, second = mujoco.MjData(source), mujoco.MjData(derived)
        first.qpos[:], second.qpos[:] = q, q
        mujoco.mj_forward(source, first)
        mujoco.mj_forward(derived, second)
        for name in ("xpos", "xmat", "xipos", "geom_xpos", "geom_xmat", "qM"):
            if not np.allclose(getattr(first, name), getattr(second, name), rtol=1e-9, atol=1e-9):
                raise ValueError(f"explicit MJCF changes pose-dependent {name}")
    return checked


def export_explicit_mjcf(source_path, output_dir):
    """Bake compiled body/joint/geom state into an independent, explicit XML."""
    import mujoco

    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    tree = ET.parse(source_path)
    root = tree.getroot()
    if root.findall(".//include"):
        raise ValueError("explicit exporter currently requires a single-file robot MJCF")
    model = mujoco.MjModel.from_xml_path(str(source_path))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = source_path.parent / compiler.get("meshdir", "")
    textdir = source_path.parent / compiler.get("texturedir", "")
    dependencies = {str(source_path): sha256(source_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(exist_ok=True)
    for tag, directory in (("mesh", meshdir), ("texture", textdir)):
        for asset in root.findall(f"asset/{tag}"):
            if asset.get("file"):
                path = (directory / asset.get("file")).resolve()
                digest = sha256(path)
                dependencies[str(path)] = digest
                cached = asset_dir / f"{digest[:16]}_{path.name}"
                if not cached.is_file() or sha256(cached) != digest:
                    shutil.copyfile(path, cached)
                dependencies[str(cached)] = digest
                # Isaac's MJCF importer incorrectly prepends the XML directory
                # to absolute paths, and writes mesh conversion temporaries next
                # to the mesh. Isolate those writes in the cache using local,
                # relative file paths; never point the importer at source assets.
                asset.set("file", cached.relative_to(output_dir).as_posix())
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    compiler.set("angle", "radian")
    # The source model uses class inheritance only on its geoms/sites. Body,
    # joint and geom physical values below are taken from the compiled model.
    for default in root.findall("default"):
        root.remove(default)
    for element in root.iter():
        element.attrib.pop("class", None)
        element.attrib.pop("childclass", None)
    orientation_keys = ("quat", "axisangle", "euler", "xyaxes", "zaxis")
    type_sizes = {"sphere": 1, "capsule": 2, "cylinder": 2, "ellipsoid": 3, "box": 3, "plane": 3}
    for body in root.findall("worldbody//body"):
        name = body.get("name")
        if not name:
            raise ValueError("explicit exporter requires named robot bodies")
        bid = model.body(name).id
        for key in orientation_keys:
            body.attrib.pop(key, None)
        body.set("pos", _numbers(model.body_pos[bid]))
        body.set("quat", _numbers(model.body_quat[bid]))
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        inertial.attrib.clear()
        inertial.attrib.update(pos=_numbers(model.body_ipos[bid]), quat=_numbers(model.body_iquat[bid]),
                               mass=_numbers([model.body_mass[bid]]), diaginertia=_numbers(model.body_inertia[bid]))
        for joint in body.findall("joint"):
            jid = model.joint(joint.get("name")).id
            for key in orientation_keys:
                joint.attrib.pop(key, None)
            kind = mujoco.mjtJoint(int(model.jnt_type[jid])).name.replace("mjJNT_", "").lower()
            joint.set("type", kind)
            joint.set("pos", _numbers(model.jnt_pos[jid]))
            joint.set("axis", _numbers(model.jnt_axis[jid]))
            joint.set("limited", str(bool(model.jnt_limited[jid])).lower())
            joint.set("range", _numbers(model.jnt_range[jid]))
            joint.set("stiffness", _numbers([model.jnt_stiffness[jid]]))
            if kind in ("hinge", "slide"):
                dof, qid = model.jnt_dofadr[jid], model.jnt_qposadr[jid]
                for key, value in (("armature", model.dof_armature[dof]), ("damping", model.dof_damping[dof]),
                                   ("frictionloss", model.dof_frictionloss[dof]), ("ref", model.qpos0[qid])):
                    joint.set(key, _numbers([value]))
        gids = np.flatnonzero(model.geom_bodyid == bid)
        geoms = body.findall("geom")
        if len(gids) != len(geoms):
            raise ValueError(f"source/compiled geom counts differ on {name}")
        for geom, gid in zip(geoms, gids):
            for key in (*orientation_keys, "fromto", "mesh", "material"):
                geom.attrib.pop(key, None)
            kind = mujoco.mjtGeom(int(model.geom_type[gid])).name.replace("mjGEOM_", "").lower()
            geom.set("type", kind)
            geom.set("pos", _numbers(model.geom_pos[gid]))
            geom.set("quat", _numbers(model.geom_quat[gid]))
            if kind in type_sizes:
                geom.set("size", _numbers(model.geom_size[gid, :type_sizes[kind]]))
            if kind == "mesh":
                geom.attrib.pop("size", None)
                mesh_id = model.geom_dataid[gid]
                geom.set("mesh", model.mesh(mesh_id).name)
                # The compiler folds mesh recentering/principal-axis rotation
                # into geom_pos/quat. Undo it before reusing the original STL,
                # otherwise compiling the derivative applies it twice.
                conjugate = model.mesh_quat[mesh_id].copy()
                conjugate[1:] *= -1
                raw_quat = np.empty(4)
                mujoco.mju_mulQuat(raw_quat, model.geom_quat[gid], conjugate)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, raw_quat)
                raw_pos = model.geom_pos[gid] - rotation.reshape(3, 3) @ model.mesh_pos[mesh_id]
                geom.set("pos", _numbers(raw_pos))
                geom.set("quat", _numbers(raw_quat))
            if model.geom_matid[gid] >= 0:
                geom.set("material", model.material(model.geom_matid[gid]).name)
            # Density of visual mesh geoms was inherited in the source. Mass is
            # explicitly defined on every body, but retain visual density=0 too.
            if not model.geom_contype[gid] and not model.geom_conaffinity[gid]:
                geom.set("density", "0")
            for key in ("contype", "conaffinity", "condim", "group", "priority"):
                geom.set(key, str(int(getattr(model, f"geom_{key}")[gid])))
            for key in ("friction", "rgba", "solref", "solimp"):
                geom.set(key, _numbers(getattr(model, f"geom_{key}")[gid]))
            for key in ("margin", "gap", "solmix"):
                geom.set(key, _numbers([getattr(model, f"geom_{key}")[gid]]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source_path.stem}_explicit.xml"
    temporary = output.with_suffix(".xml.tmp")
    ET.indent(tree, space="  ")
    try:
        tree.write(temporary, encoding="unicode", xml_declaration=False)
        derived = mujoco.MjModel.from_xml_path(str(temporary))
        checked = _equivalence(model, derived)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    provenance = dict(schema=1, source_path=str(source_path), derived_path=str(output),
                      derived_sha256=sha256(output), dependency_sha256=dependencies,
                      method="MuJoCo compiled physical values; explicit body/inertial/joint/geom; hashed relative cached assets",
                      mujoco_version=mujoco.__version__, mujoco_equivalence_verified=True,
                      verified_model_fields=checked, verified_random_fk_and_mass_matrix_poses=4,
                      isaac_importer_geometry_validated=False, isaac_dynamics_equivalence_validated=False)
    bodies = {model.body(index).name: index for index in range(1, model.nbody)}
    provenance.update(
        body_names=list(bodies),
        joint_names=[model.joint(index).name for index in range(model.njnt)
                     if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE],
        free_joint_names=[model.joint(index).name for index in range(model.njnt)
                          if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE],
        compiled_body_mass_by_name={name: float(model.body_mass[index]) for name, index in bodies.items()},
        body_inertia_diagonal_by_name={name: model.body_inertia[index].tolist() for name, index in bodies.items()},
        body_inertia_quat_wxyz_by_name={name: model.body_iquat[index].tolist() for name, index in bodies.items()},
        body_center_of_mass_local_m_by_name={name: model.body_ipos[index].tolist() for name, index in bodies.items()},
        collision_geoms=[dict(name=model.geom(index).name,
                              body=model.body(model.geom_bodyid[index]).name,
                              type=mujoco.mjtGeom(int(model.geom_type[index])).name.replace("mjGEOM_", "").lower(),
                              size=model.geom_size[index].tolist(),
                              pos_m=model.geom_pos[index].tolist(),
                              quat_wxyz=model.geom_quat[index].tolist(),
                              contype=int(model.geom_contype[index]),
                              conaffinity=int(model.geom_conaffinity[index]))
                         for index in range(model.ngeom)
                         if model.geom_contype[index] or model.geom_conaffinity[index]],
    )
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output, provenance
