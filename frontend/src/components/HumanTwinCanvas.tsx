"use client";

import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const CYAN = "#00f0ff";
const BLUE = "#0066ff";
const AMBER = "#ffaa00";
const EMERALD = "#00ff88";

export interface OrganPosition {
  id: string;
  name: string;
  system: string;
  pos: [number, number, number];
  color: string;
  targets: string[];
}

export const ANATOMICAL_ORGANS: Record<string, OrganPosition> = {
  brain: {
    id: "brain",
    name: "Brain / CNS",
    system: "Central Nervous System",
    pos: [0, 1.62, 0],
    color: "#a855f7",
    targets: ["GLP1R", "CHRNA1"],
  },
  heart: {
    id: "heart",
    name: "Heart (Myocardium)",
    system: "Cardiovascular",
    pos: [0.04, 1.18, 0.05],
    color: "#ef4444",
    targets: ["ADRB1", "CACNA1C"],
  },
  lungs: {
    id: "lung",
    name: "Lungs (Pulmonary)",
    system: "Respiratory",
    pos: [-0.07, 1.20, 0.02],
    color: "#06b6d4",
    targets: ["ACE"],
  },
  liver: {
    id: "liver",
    name: "Liver (Hepatic Parenchyma)",
    system: "Metabolic / Clearance",
    pos: [-0.06, 0.98, 0.04],
    color: "#f59e0b",
    targets: ["INSR", "HMGCR", "NR3C1"],
  },
  pancreas: {
    id: "pancreas",
    name: "Pancreas (Islets of Langerhans)",
    system: "Endocrine",
    pos: [0.03, 0.94, 0.01],
    color: "#eab308",
    targets: ["GLP1R"],
  },
  kidney: {
    id: "kidney",
    name: "Kidneys (Proximal Tubule)",
    system: "Renal / Fluid Balance",
    pos: [0.08, 0.90, -0.04],
    color: "#10b981",
    targets: ["SGLT2", "ADRB1"],
  },
  vasculature: {
    id: "vasculature",
    name: "Systemic Vasculature",
    system: "Hemodynamic SVR",
    pos: [-0.08, 0.75, 0.03],
    color: "#3b82f6",
    targets: ["AGTR1", "CACNA1C", "ACE"],
  },
  skeletal_muscle: {
    id: "skeletal_muscle",
    name: "Skeletal Muscle",
    system: "Musculoskeletal Disposal",
    pos: [0.15, 0.45, 0.02],
    color: "#8b5cf6",
    targets: ["INSR", "GLUT4", "CHRNA1"],
  },
};

function HologramShaderMaterial({
  color = CYAN,
  opacity = 0.25,
  glow = 0.4,
}: {
  color?: string;
  opacity?: number;
  glow?: number;
}) {
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
        uGlow: { value: glow },
      }}
      vertexShader={`
        varying vec3 vNormal;
        varying vec3 vPosition;
        uniform float uTime;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vPosition = mvPosition.xyz;
          float pulse = sin(position.y * 10.0 + uTime * 1.5) * 0.015;
          vec3 pos = position + normal * pulse;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `}
      fragmentShader={`
        uniform vec3 uColor;
        uniform float uOpacity;
        uniform float uGlow;
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.2);
          float scanline = sin(vPosition.y * 140.0) * 0.5 + 0.5;
          scanline = mix(0.5, 1.0, scanline);
          float alpha = (0.12 + fresnel * uGlow) * uOpacity * scanline;
          vec3 col = mix(uColor, vec3(1.0), fresnel * 0.35);
          gl_FragColor = vec4(col, alpha);
        }
      `}
    />
  );
}

function HolographicMannequin() {
  const bodyGeometry = useMemo(() => {
    const pts = [
      new THREE.Vector2(0.02, 0.0),
      new THREE.Vector2(0.09, 0.02),
      new THREE.Vector2(0.08, 0.06),
      new THREE.Vector2(0.10, 0.18),
      new THREE.Vector2(0.12, 0.34),
      new THREE.Vector2(0.15, 0.52),
      new THREE.Vector2(0.19, 0.70),
      new THREE.Vector2(0.21, 0.82),
      new THREE.Vector2(0.18, 0.95),
      new THREE.Vector2(0.20, 1.08),
      new THREE.Vector2(0.23, 1.22),
      new THREE.Vector2(0.21, 1.34),
      new THREE.Vector2(0.23, 1.42),
      new THREE.Vector2(0.16, 1.48),
      new THREE.Vector2(0.07, 1.52),
      new THREE.Vector2(0.09, 1.57),
      new THREE.Vector2(0.11, 1.64),
      new THREE.Vector2(0.09, 1.70),
      new THREE.Vector2(0.06, 1.74),
      new THREE.Vector2(0.01, 1.76),
    ];
    return new THREE.LatheGeometry(pts, 48);
  }, []);

  return (
    <group>
      <mesh geometry={bodyGeometry}>
        <HologramShaderMaterial color={CYAN} opacity={0.28} />
      </mesh>
      {/* Wireframe cage */}
      <mesh geometry={bodyGeometry}>
        <meshBasicMaterial color={CYAN} wireframe transparent opacity={0.12} />
      </mesh>
    </group>
  );
}

function OrganMarker({
  organ,
  isSelected,
  onSelect,
}: {
  organ: OrganPosition;
  isSelected: boolean;
  onSelect: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const ringRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (ringRef.current) {
      ringRef.current.rotation.z = t * 1.5;
    }
    if (pulseRef.current) {
      const s = 1.0 + 0.15 * Math.sin(t * 3.5 + organ.pos[1] * 5.0);
      pulseRef.current.scale.setScalar(s);
    }
  });

  const markerColor = isSelected ? "#ffffff" : organ.color;

  return (
    <group
      position={organ.pos}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(organ.id);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      {/* Target indicator ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.045, 0.052, 24]} />
        <meshBasicMaterial
          color={isSelected ? CYAN : markerColor}
          transparent
          opacity={isSelected ? 0.95 : hovered ? 0.8 : 0.45}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Pulsing core */}
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.024, 16, 12]} />
        <meshBasicMaterial
          color={markerColor}
          transparent
          opacity={isSelected ? 1.0 : 0.85}
        />
      </mesh>

      {/* Glow halo */}
      <mesh>
        <sphereGeometry args={[0.04, 16, 12]} />
        <meshBasicMaterial
          color={markerColor}
          transparent
          opacity={isSelected ? 0.35 : hovered ? 0.25 : 0.1}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function HudGridFloor() {
  return (
    <group position={[0, -0.02, 0]}>
      <gridHelper args={[2.5, 20, CYAN, BLUE]} position={[0, 0, 0]} />
      {/* Concentric rings */}
      {[0.4, 0.75, 1.1].map((r, i) => (
        <mesh key={i} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[r - 0.005, r, 64]} />
          <meshBasicMaterial color={CYAN} transparent opacity={0.2 - i * 0.05} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

export function HumanTwinCanvas({
  selectedOrgan,
  onSelectOrgan,
}: {
  selectedOrgan: string | null;
  onSelectOrgan: (organId: string) => void;
}) {
  return (
    <div className="w-full h-full relative cursor-grab active:cursor-grabbing">
      <Canvas
        camera={{ position: [0, 1.1, 2.3], fov: 42 }}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={["#050811"]} />
        <ambientLight intensity={0.2} />
        <pointLight position={[0, 2.5, 2.0]} intensity={0.6} color={CYAN} />
        <pointLight position={[0, 0.5, -2.0]} intensity={0.4} color={BLUE} />

        <Float speed={0.8} rotationIntensity={0.02} floatIntensity={0.04}>
          <HolographicMannequin />

          {Object.values(ANATOMICAL_ORGANS).map((organ) => (
            <OrganMarker
              key={organ.id}
              organ={organ}
              isSelected={selectedOrgan === organ.id}
              onSelect={onSelectOrgan}
            />
          ))}
        </Float>

        <HudGridFloor />

        <OrbitControls
          enablePan={false}
          minDistance={1.2}
          maxDistance={3.5}
          maxPolarAngle={Math.PI / 2 + 0.1}
          target={[0, 0.95, 0]}
        />
      </Canvas>

      {/* Viewport Overlay HUD Gridlines */}
      <div className="absolute inset-0 pointer-events-none border border-cyan-500/20 m-2 rounded-lg">
        <div className="absolute top-3 left-4 text-[10px] font-mono tracking-widest text-cyan-400/80 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping inline-block" />
          SYSTEM: WHOLE-BODY 3D DIGITAL TWIN · BIO-TELEMETRY ACTIVE
        </div>
        <div className="absolute top-3 right-4 text-[10px] font-mono text-cyan-400/60">
          SPATIAL RESOLUTION: ORGAN → REGION → CELL → MOLECULE
        </div>
        <div className="absolute bottom-3 left-4 text-[10px] font-mono text-cyan-400/60">
          ROTATION: ORBIT · CLICK NODE TO ENGAGE TARGET INSPECTOR
        </div>
        <div className="absolute bottom-3 right-4 text-[10px] font-mono text-cyan-400/60">
          COORDINATE AXIS: BI-LATERAL CORONAL / SAGITTAL
        </div>
      </div>
    </div>
  );
}

