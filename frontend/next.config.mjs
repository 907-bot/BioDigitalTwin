/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const api = process.env.NEXT_PUBLIC_API_URL || "http://api:8000";
    const ontology = process.env.NEXT_PUBLIC_ONTOLOGY_URL || "http://ontology-svc:8000";
    const knowledge = process.env.NEXT_PUBLIC_KNOWLEDGE_URL || "http://knowledge-svc:8000";
    const patient = process.env.NEXT_PUBLIC_PATIENT_URL || "http://patient-svc:8000";
    const twin = process.env.NEXT_PUBLIC_TWIN_URL || "http://twin-svc:8000";
    const agent = process.env.NEXT_PUBLIC_AGENT_URL || "http://agent-svc:8000";
    const narrative = process.env.NEXT_PUBLIC_NARRATIVE_URL || "http://narrative-svc:8000";
    return [
      // Route specific paths to microservices
      { source: "/api/ontology/:path*",  destination: `${ontology}/ontology/:path*` },
      { source: "/api/health/ontology",  destination: `${ontology}/health` },

      // Knowledge service (phases 8-15)
      { source: "/api/phase8/:path*",    destination: `${knowledge}/phase8/:path*` },
      { source: "/api/phase9/:path*",    destination: `${knowledge}/phase9/:path*` },
      { source: "/api/phase10/:path*",   destination: `${knowledge}/phase10/:path*` },
      { source: "/api/phase12/:path*",   destination: `${knowledge}/phase12/:path*` },
      { source: "/api/phase13/:path*",   destination: `${knowledge}/phase13/:path*` },
      { source: "/api/phase14/:path*",   destination: `${knowledge}/phase14/:path*` },
      { source: "/api/phase15/:path*",   destination: `${knowledge}/phase15/:path*` },
      { source: "/api/phase16/:path*",   destination: `${knowledge}/phase16/:path*` },

      // Patient service (phases 1, 2, 4)
      { source: "/api/generate-patients", destination: `${patient}/patients/generate` },
      { source: "/api/patients/:path*",  destination: `${patient}/patients/:path*` },
      { source: "/api/phase2/:path*",    destination: `${patient}/:path*` },
      { source: "/api/phase4/:path*",    destination: `${patient}/causal/:path*` },

      // Twin service (phases 3, 11, personalization)
      { source: "/api/personalization/:path*", destination: `${twin}/personalization/:path*` },
      { source: "/api/phase3/:path*",   destination: `${twin}/dynamics/:path*` },
      { source: "/api/phase11/:path*",  destination: `${twin}/phase11/:path*` },
      { source: "/api/phase16/:path*",  destination: `${twin}/phase16/:path*` },

      // Agent service (phase 5)
      { source: "/api/phase5/:path*",   destination: `${agent}/agent/:path*` },

      // Narrative service
      { source: "/api/narrative/:path*", destination: `${narrative}/narrative/:path*` },

      // Fallback: everything else to monolith API
      { source: "/api/:path*",           destination: `${api}/:path*` },
    ];
  },
};
export default nextConfig;
