"use client";
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import {
  Sphere,
  Capsule,
  Edges,
  Float,
  Text,
} from "@react-three/drei";
import * as THREE from "three";
import { ORGAN_POSITIONS, organColor, organLabel } from "./BodyGeometry";

const GLOW_CYAN = "#00d5ff";
const GLOW_BLUE = "#1a6bff";

type OrganNode = { id: string; pos: [number, number, number] };

function HolographicMaterial({ color = GLOW_CYAN, opacity = 0.25 }: { color?: string; opacity?: number }) {
  const ref = useRef<THREE.ShaderMaterial>(null);
  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.uniforms.uTime.value = clock.getElapsedTime();
    }
  });
  return (
    <shaderMaterial
      ref={ref}
      transparent
      depthWrite={false}
      uniforms={{
        uTime: { value: 0 },
        uColor: { value: new THREE.Color(color) },
        uOpacity: { value: opacity },
      }}
      vertexShader={`
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vElevation;
        uniform float uTime;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vPosition = mvPosition.xyz;
          float elevation = sin(position.y * 8.0 + uTime * 1.2) * 0.02;
          vec3 pos = position + normal * elevation;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
          vElevation = elevation;
        }
      `}
      fragmentShader={`
        uniform vec3 uColor;
        uniform float uOpacity;
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vElevation;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float fresnel = 1.0 - max(dot(viewDir, vNormal), 0.0);
          fresnel = pow(fresnel, 2.0);
          float scanline = sin(vPosition.y * 120.0 + vElevation * 10.0) * 0.5 + 0.5;
          scanline = mix(0.4, 1.0, scanline);
          float alpha = (0.15 + fresnel * 0.7) * uOpacity * scanline;
          vec3 color = mix(uColor, vec3(1.0), fresnel * 0.3);
          gl_FragColor = vec4(color, alpha);
        }
      `}
    />
  );
}

function WireframeEdges({ color = GLOW_CYAN, threshold = 15 }: { color?: string; threshold?: number }) {
  return (
    <Edges
      color={color}
      threshold={threshold}
      renderOrder={1}
    >
      <lineBasicMaterial
        color={color}
        transparent
        opacity={0.3}
      />
    </Edges>
  );
}

function HolographicOrganNode({ id, pos, color }: { id: string; pos: [number, number, number]; color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();
    const s = 1 + 0.08 * Math.sin(t * 2 + id.charCodeAt(0));
    meshRef.current.scale.setScalar(s);
  });
  return (
    <group position={pos}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.025, 12, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.04, 12, 8]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.15}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function BodySilhouette() {
  const geometry = useMemo(() => {
    const pts = [
      new THREE.Vector2(0.01, 0.00),    // feet center
      new THREE.Vector2(0.10, 0.02),    // feet
      new THREE.Vector2(0.08, 0.05),    // ankle
      new THREE.Vector2(0.11, 0.15),    // calf
      new THREE.Vector2(0.13, 0.30),    // knee
      new THREE.Vector2(0.16, 0.45),    // thigh
      new THREE.Vector2(0.20, 0.60),    // upper thigh
      new THREE.Vector2(0.22, 0.75),    // hip
      new THREE.Vector2(0.20, 0.85),    // waist
      new THREE.Vector2(0.18, 0.95),    // lower waist
      new THREE.Vector2(0.22, 1.05),    // mid torso
      new THREE.Vector2(0.24, 1.15),    // chest
      new THREE.Vector2(0.22, 1.30),    // shoulders start
      new THREE.Vector2(0.24, 1.40),    // shoulders
      new THREE.Vector2(0.17, 1.45),    // shoulders to neck
      new THREE.Vector2(0.07, 1.50),    // neck
      new THREE.Vector2(0.10, 1.55),    // lower head
      new THREE.Vector2(0.11, 1.62),    // mid head
      new THREE.Vector2(0.10, 1.68),    // upper head
      new THREE.Vector2(0.07, 1.73),    // top of head
      new THREE.Vector2(0.01, 1.75),    // crown
    ];
    return new THREE.LatheGeometry(pts, 48);
  }, []);

  return (
    <mesh geometry={geometry}>
      <HolographicMaterial color={GLOW_CYAN} opacity={0.3} />
      <WireframeEdges color={GLOW_CYAN} threshold={15} />
    </mesh>
  );
}

function BodyArms() {
  // Arms positioned at shoulder height, slightly away from body
  const armPositions = [
    { x: -0.30, y: 1.38, z: 0.00, rot: 0.25 },
    { x: 0.30, y: 1.38, z: 0.00, rot: -0.25 },
  ];
  return (
    <>
      {armPositions.map((pos, i) => (
        <group key={i}>
          {/* upper arm */}
          <Capsule args={[0.04, 0.28, 8, 12]}
            position={[pos.x, pos.y - 0.14, pos.z]}
            rotation={[0, 0, pos.rot]}>
            <HolographicMaterial color={GLOW_CYAN} opacity={0.25} />
            <WireframeEdges color={GLOW_CYAN} threshold={15} />
          </Capsule>
          {/* forearm */}
          <Capsule args={[0.035, 0.26, 8, 12]}
            position={[pos.x * 1.2, pos.y - 0.38, pos.z * 0.5]}
            rotation={[0.15, 0, pos.rot * 0.5]}>
            <HolographicMaterial color={GLOW_CYAN} opacity={0.25} />
            <WireframeEdges color={GLOW_CYAN} threshold={15} />
          </Capsule>
        </group>
      ))}
    </>
  );
}

function GridFloor() {
  return (
    <group position={[0, 0.01, 0]}>
      {Array.from({ length: 11 }).map((_, i) => (
        <gridHelper
          key={i}
          args={[3, 12, GLOW_CYAN, GLOW_BLUE]}
          position={[0, 0, 0]}
          rotation={[Math.PI / 2, 0, (i * Math.PI) / 5.5]}
        />
      ))}
    </group>
  );
}

export function HolographicBody() {
  return (
    <Float
      speed={0.5}
      rotationIntensity={0.01}
      floatIntensity={0.03}
    >
      <group>
        <BodySilhouette />
        <BodyArms />
      </group>
    </Float>
  );
}

export function HolographicScene({ selected }: { selected: string | null }) {
  const organs: OrganNode[] = useMemo(() => {
    return Object.entries(ORGAN_POSITIONS).map(([id, pos]) => ({
      id,
      pos: pos as [number, number, number],
    }));
  }, []);

  return (
    <>
      <ambientLight intensity={0.1} />
      <pointLight position={[0, 2, 2]} intensity={0.3} color={GLOW_CYAN} />
      <pointLight position={[0, 1, -2]} intensity={0.2} color={GLOW_BLUE} />

      <HolographicBody />

      {organs.map((org) => {
        const isSelected = selected === org.id;
        const color = organColor(org.id, GLOW_CYAN);
        return (
          <HolographicOrganNode
            key={org.id}
            id={org.id}
            pos={org.pos}
            color={isSelected ? "#ffffff" : color}
          />
        );
      })}

      <GridFloor />
    </>
  );
}
