const j = async (res) => {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
};

export const getHealth = () => fetch("/api/health").then(j);
export const getDatasets = () => fetch("/api/datasets").then(j);
export const runPipeline = (body) =>
  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(j);
export const getOpenmpBenchmarks = () => fetch("/api/benchmarks/openmp").then(j);
export const getMpiBenchmarks = () => fetch("/api/benchmarks/mpi").then(j);
