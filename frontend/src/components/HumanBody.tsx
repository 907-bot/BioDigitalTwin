"use client";
import { Sphere, Capsule, Cylinder } from "@react-three/drei";
import { useBodyGeometry } from "./BodyGeometry";

/**
 * Stylized low-poly human body built from primitives.
 * All measurements in meters. y=0 is feet, y=1.8 is head.
 */
export function HumanBody() {
  const g = useBodyGeometry();
  const skin = "#f1c4a0";
  const skinDark = "#d9a98a";
  const cloth = "#1f3a5f";
  const clothAccent = "#3b6098";

  return (
    <group>
      {/* head */}
      <Sphere args={[g.head.r, 24, 16]} position={[0, g.head.y, 0]}>
        <meshStandardMaterial color={skin} roughness={0.7} />
      </Sphere>
      {/* neck */}
      <Cylinder args={[g.neck.rTop, g.neck.rBot, g.neck.h, 16]} position={[0, g.neck.y, 0]}>
        <meshStandardMaterial color={skin} roughness={0.7} />
      </Cylinder>
      {/* torso (chest capsule) */}
      <Capsule args={[0.15, 0.50, 8, 16]} position={[0, g.torso.y, 0]}>
        <meshStandardMaterial color={cloth} roughness={0.6} />
      </Capsule>
      {/* shirt collar accent (thinner capsule) */}
      <Cylinder args={[0.18, 0.18, 0.05, 24]} position={[0, g.neck.y - 0.07, 0]}>
        <meshStandardMaterial color={clothAccent} roughness={0.6} />
      </Cylinder>
      {/* shoulders + arms */}
      {[-1, 1].map((s) => (
        <group key={s}>
          {/* upper arm */}
          <Capsule
            args={[0.07, g.shoulderL.len, 6, 12]}
            position={[s * 0.28, g.shoulderL.y, 0]}
            rotation={[0, 0, s * 0.18]}
          >
            <meshStandardMaterial color={cloth} roughness={0.6} />
          </Capsule>
          {/* forearm */}
          <Capsule
            args={[0.06, g.armL.len, 6, 12]}
            position={[s * 0.34, g.armL.y, 0]}
            rotation={[0, 0, s * 0.05]}
          >
            <meshStandardMaterial color={skin} roughness={0.7} />
          </Capsule>
          {/* hand */}
          <Sphere args={[0.06, 12, 8]} position={[s * 0.36, g.armL.y - 0.30, 0]}>
            <meshStandardMaterial color={skin} roughness={0.8} />
          </Sphere>
        </group>
      ))}
      {/* hip */}
      <Capsule args={[0.08, 0.10, 8, 16]} position={[0, g.hip.y, 0]}>
        <meshStandardMaterial color={clothAccent} roughness={0.6} />
      </Capsule>
      {/* legs */}
      {[-1, 1].map((s) => (
        <Capsule
          key={s}
          args={[0.09, g.legL.len, 6, 12]}
          position={[s * 0.10, g.legL.y, 0]}
        >
          <meshStandardMaterial color={clothAccent} roughness={0.6} />
        </Capsule>
      ))}
      {/* feet */}
      {[-1, 1].map((s) => (
        <Capsule
          key={`f${s}`}
          args={[0.07, 0.10, 4, 12]}
          position={[s * 0.10, 0.05, 0.08]}
          rotation={[Math.PI / 2.2, 0, 0]}
        >
          <meshStandardMaterial color={skinDark} roughness={0.8} />
        </Capsule>
      ))}
    </group>
  );
}
