"""Small articulated robots for deterministic game-mechanics tests."""

from __future__ import annotations

from pathlib import Path


def write_microduck_fixture(tmp_path: Path) -> Path:
    """A 14-actuator stand-in for the MicroDuck MJCF that rests on two feet."""
    lines = [
        '<mujoco model="microduck-test">',
        '  <option timestep="0.002"/>',
        (
            '  <default><joint damping="0.1" limited="true" range="-1 1"/>'
            '<geom density="1000" contype="0" conaffinity="0"/></default>'
        ),
        '  <worldbody><geom type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>',
        '    <body name="torso" pos="0 0 0.12"><freejoint name="root"/>',
        '      <geom type="sphere" size="0.02" mass="0.1"/>',
        '      <site name="head_imu" pos="0 0 0.02" quat="0.707107 0 -0.707107 0"/>',
        '      <site name="mouth_tip" pos="0.0266783 0 -0.00332564"/>',
        '      <body name="camera_mount" pos="0.026 0 0.02" '
        'quat="0.707107 0 -0.707107 0">',
        '        <site name="head_camera" quat="0.707107 0 0.707107 0"/>',
        '        <camera name="head_camera" quat="0 0 -1 0" fovy="90"/>',
        "      </body>",
    ]
    for side, y in (("left", 0.03), ("right", -0.03)):
        lines.extend(
            (
                f'      <body name="{side}_foot_body" pos="0 {y} -0.11">',
                f'        <geom name="{side}_foot_collision" type="box" '
                'size="0.02 0.01 0.005" contype="1" conaffinity="1" mass="0.02"/>',
                f'        <site name="{side}_foot" pos="0 0 -0.005"/>',
                "      </body>",
            )
        )
    for index in range(14):
        axis = ("1", "0", "0") if index % 2 == 0 else ("0", "1", "0")
        indent = "      " + "  " * index
        lines.extend(
            (
                f'{indent}<body name="link{index}">',
                f'{indent}  <joint name="joint{index}" axis="{" ".join(axis)}"/>',
                f'{indent}  <geom type="sphere" size="0.005" mass="0.01"/>',
            )
        )
    for index in reversed(range(14)):
        lines.append("      " + "  " * index + "</body>")
    stand_qpos = "0 0 0.12 1 0 0 0 " + " ".join(["0"] * 14)
    stand_ctrl = " ".join(["0"] * 14)
    lines.extend(
        (
            "    </body>",
            "  </worldbody>",
            "  <actuator>",
            *(
                f'    <position name="actuator{index}" joint="joint{index}" '
                'kp="5" ctrlrange="-1 1"/>'
                for index in range(14)
            ),
            "  </actuator>",
            "  <keyframe>",
            f'    <key name="STAND" qpos="{stand_qpos}" ctrl="{stand_ctrl}"/>',
            "  </keyframe>",
            "</mujoco>",
        )
    )
    scene = tmp_path / "scene_walk.xml"
    scene.write_text("\n".join(lines))
    return scene
