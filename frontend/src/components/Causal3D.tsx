"use client";
import { useMemo, useState, useRef, useEffect, Suspense } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Line, Text } from "@react-three/drei";
import { EffectComposer, Bloom, ChromaticAberration } from "@react-three/postprocessing";
import * as THREE from "three";
import { HolographicBody } from "./HolographicBody";
import { organPosition, organColor, organLabel, ORGAN_POSITIONS } from "./BodyGeometry";
import type { CausalGraph } from "@/lib/api";

const KIND_COLOR: Record<string, string> = {
  biomarker:   "#1D9E75",
  organ:       "#7F77DD",
  disease:     "#D85A30",
  demographic: "#BA7517",
};

const GLOW_CYAN = "#00d5ff";

type Edge = { src: string; dst: string; weight: number; rel?: string };

function OrganNode({ id, pos, kind, label, color, highlighted, dim, onClick }: {
  id: string; pos: [number, number, number]; kind: string; label: string;
  color: string; highlighted: boolean; dim: boolean; onClick: (id: string) => void;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const [hover, setHover] = useState(false);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    if (highlighted) {
      const s = 1 + 0.12 * Math.sin(t * 4);
      ref.current.scale.setScalar(s);
    } else {
      ref.current.scale.setScalar(1);
    }
  });
  return (
    <group position={pos}>
      <mesh
        ref={ref}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
        onClick={() => onClick(id)}
      >
        <sphereGeometry args={[hover || highlighted ? 0.045 : 0.035, 16, 12]} />
        <meshBasicMaterial
          color={dim ? "#475569" : color}
          transparent
          opacity={dim ? 0.3 : 0.9}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[0.05, 12, 8]} />
        <meshBasicMaterial
          color={highlighted ? "#ffffff" : color}
          transparent
          opacity={highlighted ? 0.3 : 0.1}
          depthWrite={false}
        />
      </mesh>
      {(hover || highlighted) && (
        <Html
          position={[0, 0.06, 0]}
          center
          style={{ pointerEvents: "none" }}
          distanceFactor={1.4}
        >
          <div className="bg-bg/95 text-text text-[10px] px-1.5 py-0.5 rounded border border-border whitespace-nowrap">
            <span className="text-muted mr-1">{kind}</span>
            <span className="font-mono">{label}</span>
          </div>
        </Html>
      )}
    </group>
  );
}

function EdgeArc({ from, to, highlighted, dim }: {
  from: [number, number, number]; to: [number, number, number];
  highlighted: boolean; dim: boolean;
}) {
  const mid: [number, number, number] = [
    (from[0] + to[0]) / 2,
    (from[1] + to[1]) / 2,
    (from[2] + to[2]) / 2 + 0.04,
  ];
  const points = [from, mid, to] as [number, number, number][];
  const color = dim ? "#1e293b" : highlighted ? "#fbbf24" : "#00d5ff";
  return (
    <Line
      points={points}
      color={color}
      lineWidth={highlighted ? 2.5 : 0.8}
      transparent
      opacity={dim ? 0.1 : 0.6}
    />
  );
}

function HolographicEffects() {
  return (
    <EffectComposer>
      <Bloom
        luminanceThreshold={0}
        luminanceSmoothing={0.9}
        intensity={0.8}
        mipmapBlur
      />
      <ChromaticAberration
        offset={new THREE.Vector2(0.001, 0.001)}
      />
    </EffectComposer>
  );
}

function CausalScene({ graph, selected, onSelect }: {
  graph: CausalGraph; selected: string | null; onSelect: (id: string | null) => void;
}) {
  const { positions, edges } = useMemo(() => {
    const pos: Record<string, [number, number, number]> = {};
    graph.nodes.forEach((n) => {
      const p = organPosition(n.id);
      if (p) pos[n.id] = p;
    });
    const e: Edge[] = graph.edges
      .filter((ed: any) => pos[ed.src] && pos[ed.dst])
      .map((ed: any) => ({ src: ed.src, dst: ed.dst, weight: ed.weight, rel: ed.rel }));
    return { positions: pos, edges: e };
  }, [graph]);

  const connected = useMemo(() => {
    if (!selected) return new Set<string>();
    const s = new Set<string>([selected]);
    edges.forEach((e) => {
      if (e.src === selected) s.add(e.dst);
      if (e.dst === selected) s.add(e.src);
    });
    return s;
  }, [selected, edges]);

  const orphanNodes = graph.nodes.filter((n) => !positions[n.id]);

  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 2, 3]} intensity={0.4} color={GLOW_CYAN} />
      <pointLight position={[0, 1, -2]} intensity={0.2} color="#1a6bff" />

      <Suspense fallback={null}>
        <HolographicBody />
      </Suspense>

      {/* edges */}
      {edges.map((e, i) => {
        const f = positions[e.src];
        const t = positions[e.dst];
        if (!f || !t) return null;
        const hi = !!selected && (e.src === selected || e.dst === selected);
        return (
          <EdgeArc key={i} from={f} to={t} highlighted={hi}
                   dim={!!selected && !hi} />
        );
      })}

      {/* nodes */}
      {graph.nodes.map((n) => {
        const p = positions[n.id];
        if (!p) return null;
        const hi = !!selected && (selected === n.id || connected.has(n.id));
        const dim = !!selected && !hi;
        const color = organColor(n.id, KIND_COLOR[n.kind] || "#1f2937");
        return (
          <OrganNode
            key={n.id}
            id={n.id}
            pos={p}
            kind={n.kind}
            label={organLabel(n.id)}
            color={color}
            highlighted={selected === n.id}
            dim={dim}
            onClick={(id) => onSelect(selected === id ? null : id)}
          />
        );
      })}

      {orphanNodes.length > 0 && (
        <Text
          position={[0, 2.1, 0]}
          fontSize={0.05}
          color="#94a3b8"
          anchorX="center"
        >
          {orphanNodes.length} node(s) unmapped to anatomy
        </Text>
      )}
    </>
  );
}

export function Causal3D({ graph, height = 560 }: {
  graph: CausalGraph; height?: number;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const sel = selected ? graph.nodes.find((n) => n.id === selected) : null;

  useEffect(() => { setSelected(null); }, [graph]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_240px] gap-4">
      <div className="rounded-md overflow-hidden border border-border bg-gradient-to-b from-slate-900 to-slate-950"
           style={{ height }}>
        <Canvas
          shadows
          camera={{ position: [0, 1.0, 2.2], fov: 40 }}
          dpr={[1, 2]}
        >
          <color attach="background" args={["#05080f"]} />
          <CausalScene graph={graph} selected={selected} onSelect={setSelected} />
          <HolographicEffects />
          <OrbitControls
            enablePan
            enableZoom
            enableRotate
            minDistance={0.8}
            maxDistance={4.5}
            target={[0, 1.0, 0]}
          />
        </Canvas>
      </div>
      <div className="space-y-3">
        <div>
          <div className="label">3D holographic view</div>
          <p className="text-[10px] text-muted mt-1">
            Drag to rotate · scroll to zoom · click an organ node to
            highlight its causal connections
          </p>
        </div>
        {sel ? (
          <div className="rounded border border-border bg-bg p-3">
            <div className="text-[10px] text-muted uppercase">{sel.kind}</div>
            <div className="text-text text-sm font-mono mt-0.5">
              {organLabel(sel.id)}
            </div>
            <div className="text-[10px] text-muted mt-1">
              {Object.entries(sel).filter(([k]) => k !== "id" && k !== "kind").slice(0, 4).map(([k, v]) => (
                <div key={k}>
                  <span className="text-muted">{k}:</span>{" "}
                  <span className="text-text font-mono">{String(v)}</span>
                </div>
              ))}
            </div>
            <button
              onClick={() => setSelected(null)}
              className="mt-2 text-[10px] text-teal hover:underline"
            >
              clear selection
            </button>
          </div>
        ) : (
          <div className="rounded border border-border bg-bg p-3 text-xs text-muted">
            A holographic 3D body with organ nodes and causal
            connections shown inside.
          </div>
        )}
        <div>
          <div className="label">Legend</div>
          <div className="space-y-1 mt-1">
            {Array.from(new Set(graph.nodes.map((n) => n.kind))).map((k) => (
              <div key={k} className="flex items-center gap-2 text-[10px]">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ background: KIND_COLOR[k] || "#1f2937" }}
                />
                <span className="text-muted">{k}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="text-[10px] text-muted leading-relaxed pt-2 border-t border-border">
          {Object.keys(ORGAN_POSITIONS).length} anatomical landmarks on
          a stylized holographic body model.
        </div>
      </div>
    </div>
  );
}
