// An system-wide progress page, that gives an overview of which workspaces are running, what resources they use, etc.
import { Link } from "react-router";
import Back from "~icons/tabler/arrow-left";
import ScaleUp from "~icons/tabler/arrow-up";
import ScaleDown from "~icons/tabler/arrow-down";
import Pause from "~icons/tabler/player-pause-filled";
import Play from "~icons/tabler/player-play-filled";
import Stop from "~icons/tabler/player-stop-filled";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";
import { useState } from "react";

function timeLeft(workspace: any) {
  if (!workspace.resources.gpus) {
    return "paused";
  }
  const now = Date.now();
  const eta = workspace.eta;
  const diff = eta - now;

  if (diff <= 0) {
    return "Done";
  }

  const minutes = Math.floor(diff / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);

  return `${minutes}m ${seconds}s left`;
}

export default function ProgressPage() {
  const [currentTab, setCurrentTab] = useState("workspaces");
  const tabs = [
    { id: "workspaces", label: "Running workspaces" },
    { id: "nims", label: "Running NIMs" },
    { id: "users", label: "Users & groups" },
  ];

  // Mock data for now.
  const [data, setData] = useState({
    workspaces: [
      {
        name: "Drug discovery pipeline",
        user: "Derek Smith",
        start_time: Date.now() - 1000 * 60 * 60 * 2,
        eta: Date.now() + 1000 * 60 * 34.3,
        boxes_done: 18,
        total_boxes: 24,
        current_box_progress: 0.45,
        overall_progress: 0.75,
        resources: { gpus: 8 },
      },
      {
        name: "Generating receptor conformations",
        user: "Derek Smith",
        start_time: Date.now() - 1000 * 60 * 45,
        eta: Date.now() + 1000 * 60 * 21.1,
        boxes_done: 3,
        total_boxes: 12,
        current_box_progress: 0.6,
        overall_progress: 0.28,
        resources: { gpus: 2 },
      },
      {
        name: "Training docking model",
        user: "Dora Gera",
        start_time: Date.now() - 1000 * 60 * 10,
        eta: Date.now() + 1000 * 60 * 122.5,
        boxes_done: 1,
        total_boxes: 20,
        current_box_progress: 0.05,
        overall_progress: 0.05,
        resources: { gpus: 4 },
      },
      {
        name: "Side effects and toxicity screening",
        user: "Rajat Kumar Pal",
        start_time: Date.now() - 1000 * 60 * 60 * 6,
        eta: Date.now() + 1000 * 60 * 60 * 1.77,
        boxes_done: 40,
        total_boxes: 50,
        current_box_progress: 0.9,
        overall_progress: 0.82,
        resources: { gpus: 6 },
      },
      {
        name: "Molecular dynamics simulation",
        user: "Rajat Kumar Pal",
        start_time: Date.now() - 1000 * 60 * 60 * 24,
        eta: Date.now() + 1000 * 60 * 60 * 12.1,
        boxes_done: 120,
        total_boxes: 200,
        current_box_progress: 0.4,
        overall_progress: 0.6,
        resources: { gpus: 16 },
      },
      {
        name: "MicroRNA experiment",
        user: "Livia Babos",
        start_time: Date.now() - 1000 * 60 * 15,
        eta: Date.now() + 1000 * 60 * 45.8,
        boxes_done: 2,
        total_boxes: 8,
        current_box_progress: 0.25,
        overall_progress: 0.2,
        resources: { gpus: 1 },
      },
    ],
    nims: [
      { publisher: "MIT", name: "Boltz-2", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      { publisher: "Colabfold", name: "msa-search", status: "running", replicasHealthy: 10, replicasRequested: 10 },
      { publisher: "Openfold", name: "openfold2", status: "running", replicasHealthy: 1, replicasRequested: 1 },
      { publisher: "Arc", name: "evo2-40b", status: "running", replicasHealthy: 0, replicasRequested: 3 },
      { publisher: "NVIDIA", name: "genmol", status: "running", replicasHealthy: 10, replicasRequested: 1 },
      { publisher: "DeepMind", name: "alphafold2-multimer", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      { publisher: "Meta", name: "esm2-650m", status: "running", replicasHealthy: 12, replicasRequested: 12 },
      // { publisher: "DeepMind", name: "alphafold2", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      // { publisher: "IPD", name: "ProteinMPNN", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      // { publisher: "IPD", name: "rfdiffusion", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      // { publisher: "NVIDIA", name: "MolMIM", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      // { publisher: "Meta", name: "esmfold", status: "running", replicasHealthy: 3, replicasRequested: 3 },
      // { publisher: "MIT", name: "diffdock", status: "running", replicasHealthy: 3, replicasRequested: 3 },
    ]
  });

  function scaleNIM(nim: any, newReplicaCount: number) {
    // For now, just update the mock data. In a real implementation, this would make an API call.
    setData((prevData) => {
      const newNims = prevData.nims.map((n) => {
        if (n.name === nim.name) {
          return { ...n, replicasRequested: newReplicaCount };
        }
        return n;
      });
      return { ...prevData, nims: newNims };
    });
  }

  function setResources(ws: any, resources: any) {
    // For now, just update the mock data. In a real implementation, this would make an API call.
    setData((prevData) => {
      const newWorkspaces = prevData.workspaces.map((w) => {
        if (w.name === ws.name) {
          return { ...w, resources, prevResources: w.resources };
        }
        return w;
      });
      return { ...prevData, workspaces: newWorkspaces };
    });
  }


  return (
    <div className="progress-page">
      <div className="logo">
        <a href="https://lynxkite.com/">
          <img src={logo} className="logo-image" alt="LynxKite logo" />
        </a>
        <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
        <div className="tagline">The Complete Graph Data Science Platform</div>
      </div>
      <div className="entry-list">

        <div role="tablist" className="tabs tabs-border">
          <div className="tab"><Link to="/"><Back /></Link></div>
          {tabs.map(tab => (
            <a key={tab.id} role="tab" onClick={() => setCurrentTab(tab.id)}
              className={currentTab === tab.id ? "tab tab-active" : "tab"}>{tab.label}</a>
          ))}
        </div>

        {currentTab === "workspaces" && <Workspaces workspaces={data.workspaces} setResources={setResources} />}
        {currentTab === "nims" && <NIMs nims={data.nims} scaleNIM={scaleNIM} />}
      </div>
    </div>
  );
}

function Workspaces(props: { workspaces: any[], setResources: (ws: any, resources: any) => void }) {
  if ((props.workspaces?.length ?? 0) === 0) {
    return <div>No workspaces in progress.</div>;
  }
  return <table className="progress-table">
    <thead>
      <tr>
        <th>Workspace</th>
        <th>User</th>
        <th colSpan={2}>Progress</th>
        <th>GPUs</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {props.workspaces.map((ws) => (
        <tr key={ws.name}>
          <td className="workspace-name">{ws.name}</td>
          <td className="workspace-user">{ws.user}</td>
          <td className="workspace-progress">
            <progress className={`progress progress-${ws.resources.gpus ? "primary" : "secondary"} w-50`}
            value={ws.overall_progress * 100} max="100"></progress>
          </td>
          <td className="workspace-eta">
            {timeLeft(ws)}
          </td>
          <td className="workspace-resources">{ws.resources.gpus}</td>
          <td className="table-actions">
            {ws.resources.gpus ? (
              <button className="btn btn-sm" title="Pause" onClick={() => props.setResources(ws, {})}><Pause /></button>
            ) : (
              <button className="btn btn-sm" title="Resume" onClick={() => props.setResources(ws, { gpus: 1 })}><Play /></button>
            )}
            <button className="btn btn-sm" title="Stop"><Stop /></button>
          </td>
        </tr>
      ))}
    </tbody>
  </table>;
}


function NIMs(props: { nims: any[], scaleNIM: (nim: any, replicas: number) => void }) {
  if ((props.nims?.length ?? 0) === 0) {
    return <div>No NIMs deployed.</div>;
  }
  return <table className="progress-table">
    <thead>
      <tr>
        <th>NIM</th>
        <th>Replicas</th>
        <th>Status</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {props.nims.map((nim) => (
        <tr key={nim.name}>
          <td className="nim-name">{nim.name} <span className="nim-publisher">from {nim.publisher}</span></td>
          <td className="nim-replica-count">{nim.replicasHealthy}{nim.replicasHealthy !== nim.replicasRequested && ` / ${nim.replicasRequested}`}</td>
          <td className="nim-status"> {nim.replicasHealthy !== nim.replicasRequested ? 'resizing' : nim.status} </td>
          <td className="table-actions">
            <button className="btn btn-sm" title="Scale up" onClick={() => props.scaleNIM(nim, nim.replicasRequested + 1)}><ScaleUp /></button>
            <button className="btn btn-sm" title="Scale down" onClick={() => props.scaleNIM(nim, nim.replicasRequested - 1)}><ScaleDown /></button>
            <button className="btn btn-sm" title="Stop" onClick={() => props.scaleNIM(nim, 0)}><Stop /></button>
          </td>
        </tr>
      ))}
    </tbody>
  </table>;
}
