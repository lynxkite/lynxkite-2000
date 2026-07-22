// The LynxKite workspace editor.

import {
  Background,
  BackgroundVariant,
  type Connection,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type XYPosition,
} from "@xyflow/react";
import axios from "axios";
import {
  lazy,
  type MouseEvent,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link } from "react-router";
import useSWR, { type Fetcher } from "swr";
import type { Array as YArray, Map as YMap } from "yjs";
import Arrow from "~icons/tabler/arrow-wave-right-up.jsx";
import Backspace from "~icons/tabler/backspace.jsx";
import GridDots from "~icons/tabler/grid-dots.jsx";
import LibraryMinus from "~icons/tabler/library-minus.jsx";
import LibraryPlus from "~icons/tabler/library-plus.jsx";
import Pause from "~icons/tabler/player-pause.jsx";
import Play from "~icons/tabler/player-play.jsx";
import Robot from "~icons/tabler/robot.jsx";
import RotateClockwise from "~icons/tabler/rotate-clockwise.jsx";
import Transfer from "~icons/tabler/transfer.jsx";
import Close from "~icons/tabler/x.jsx";
import type { Op as OpsOp, WorkspaceNode } from "../apiTypes.ts";
import favicon from "../assets/favicon.ico";
import {
  apiJson,
  getConfig,
  parentPath,
  uploadFile,
  useFolderPermissions,
  usePath,
} from "../common.ts";
import Tooltip from "../Tooltip.tsx";
import UserMenu from "../UserMenu";
import { useAutoConnect } from "./autoConnect.ts";
import { copySelection, cutSelection, pasteSelection } from "./clipboard.ts";
import { nodeToYMap, useCRDTWorkspace } from "./crdt.ts";
import EnvironmentSelector from "./EnvironmentSelector";
import ExecutionOptions from "./ExecutionOptions.tsx";
import { snapChangesToGrid } from "./grid.ts";
import LynxKiteEdge from "./LynxKiteEdge.tsx";
import { LynxKiteState } from "./LynxKiteState";
import NodeSearch, { buildCategoryHierarchy, type Catalogs } from "./NodeSearch.tsx";
import NodeWithGraphCreationView from "./nodes/GraphCreationNode.tsx";
import Group from "./nodes/Group.tsx";
import NodeWithComment from "./nodes/NodeWithComment.tsx";
import NodeWithGradio from "./nodes/NodeWithGradio.tsx";
import NodeWithImage from "./nodes/NodeWithImage.tsx";
import NodeWithMolecule from "./nodes/NodeWithMolecule.tsx";
import NodeWithParams from "./nodes/NodeWithParams";
import NodeWithTableView from "./nodes/NodeWithTableView.tsx";
import NodeWithVisualization from "./nodes/NodeWithVisualization.tsx";
import { WorkspaceProgress } from "./WorkspaceProgress.tsx";

const Assistant = lazy(() => import("./Assistant.tsx"));

// The workspace gets re-rendered on every frame when a node is moved.
// Surprisingly, re-rendering the icons is very expensive in dev mode.
// Memoizing them fixes it.
const DeleteIcon = memo(Backspace);
const GridIcon = memo(GridDots);
const GridOffIcon = memo(Arrow);
const GroupIcon = memo(LibraryPlus);
const UngroupIcon = memo(LibraryMinus);
const RestartIcon = memo(RotateClockwise);
const RobotIcon = memo(Robot);
const PlayIcon = memo(Play);
const PauseIcon = memo(Pause);
const CloseIcon = memo(Close);
const ChangeTypeIcon = memo(Transfer);

export default function Workspace(props: any) {
  return (
    <ReactFlowProvider>
      <LynxKiteFlow {...props} />
    </ReactFlowProvider>
  );
}

const ICONIZE_THRESHOLD = 0.3;

function LynxKiteFlow() {
  const reactFlow = useReactFlow();
  const reactFlowContainer = useRef<HTMLDivElement>(null);
  const cursorScreenPos = useRef<XYPosition | null>(null);
  const [isShiftPressed, setIsShiftPressed] = useState(false);
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);
  const [gridSnapEnabled, setGridSnapEnabled] = useState(
    () => localStorage.getItem("gridSnapEnabled") === "true",
  );
  const path = usePath().replace(/^[/]edit[/]/, "");
  const [message, setMessage] = useState(null as string | null);
  const [iconized, setIconized] = useState(reactFlow.getZoom() < ICONIZE_THRESHOLD);
  const shortPath = path!
    .split("/")
    .pop()!
    .replace(/[.]lynxkite[.]json$/, "");
  const permissions = useFolderPermissions(path);
  const canWrite = permissions.write;
  const crdt = useCRDTWorkspace(path, canWrite);
  const nodes = crdt.feNodes;
  const edges = crdt.feEdges;
  const autoConnect = useAutoConnect(edges, crdt);

  // Track Shift key state
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Shift") {
        setIsShiftPressed(true);
      }
    }

    function handleKeyUp(event: KeyboardEvent): void {
      if (event.key === "Shift") {
        setIsShiftPressed(false);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem("gridSnapEnabled", String(gridSnapEnabled));
  }, [gridSnapEnabled]);

  const fetcher: Fetcher = (resource: string, init?: RequestInit) => apiJson(resource, init);
  const encodedPathForAPI = path!
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const catalog = useSWR<Catalogs>(
    `/api/catalog?workspace=${encodedPathForAPI}`,
    fetcher as Fetcher<Catalogs>,
  );
  const config = getConfig();
  const categoryHierarchy = useMemo(() => {
    if (!catalog.data || !crdt?.ws?.env) return undefined;
    return buildCategoryHierarchy(catalog.data[crdt.ws.env]);
  }, [catalog, crdt]);
  const [suppressSearchUntil, setSuppressSearchUntil] = useState(0);
  const [nodeSearchSettings, setNodeSearchSettings] = useState(
    undefined as
      | {
          pos: XYPosition;
        }
      | undefined,
  );
  const nodeTypes = useMemo(
    () => ({
      basic: NodeWithParams,
      visualization: NodeWithVisualization,
      image: NodeWithImage,
      table_view: NodeWithTableView,
      service: NodeWithTableView,
      gradio: NodeWithGradio,
      graph_creation_view: NodeWithGraphCreationView,
      molecule: NodeWithMolecule,
      comment: NodeWithComment,
      node_group: Group,
    }),
    [],
  );
  const edgeTypes = useMemo(
    () => ({
      default: LynxKiteEdge,
    }),
    [],
  );

  function clearSelection() {
    if (!crdt) return;
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => e.selected);
    if (selectedNodes.length > 0) {
      crdt.onFENodesChange?.(
        selectedNodes.map((n) => ({ id: n.id, type: "select" as const, selected: false })),
      );
    }
    if (selectedEdges.length > 0) {
      crdt.onFEEdgesChange?.(
        selectedEdges.map((e) => ({ id: e.id, type: "select" as const, selected: false })),
      );
    }
  }

  // Global keyboard shortcuts.
  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const isPrimaryModifierPressed = event.ctrlKey || event.metaKey;
      // Show the node search dialog on "/".
      if (nodeSearchSettings || isTypingInFormElement()) return;
      if (event.key === "/" && categoryHierarchy && canWrite) {
        event.preventDefault();
        setNodeSearchSettings({
          pos: getBestPosition(),
        });
      } else if (event.key === "Escape") {
        event.preventDefault();
        clearSelection();
      } else if (event.key === "r" && canWrite) {
        event.preventDefault();
        executeWorkspace();
      } else if (isPrimaryModifierPressed) {
        if (event.key === "z" && canWrite) {
          crdt?.undo();
        } else if (event.key === "y" && canWrite) {
          crdt?.redo();
        } else if (!(nodeSearchSettings || isTypingInFormElement())) {
          const key = event.key.toLowerCase();
          if (key === "c") {
            copySelection(nodes, edges, crdt?.ws ?? {});
          } else if (key === "v" && canWrite) {
            pasteSelection(crdt, cursorScreenPos, reactFlow, setMessage);
          } else if (key === "x" && canWrite) {
            cutSelection(nodes, edges, crdt?.ws ?? {}, deleteSelection);
          } else if (key === "a") {
            event.preventDefault();
            selectAll();
          }
        }
      }
    };
    document.addEventListener("keyup", handleKey);
    return () => {
      document.removeEventListener("keyup", handleKey);
    };
  }, [categoryHierarchy, nodeSearchSettings, canWrite, crdt, nodes, edges]);

  function getBestPosition() {
    const W = reactFlowContainer.current!.clientWidth;
    const H = reactFlowContainer.current!.clientHeight;
    const w = 200;
    const h = 200;
    const SPEED = 20;
    const GAP = 50;
    const pos = { x: 100, y: 100 };
    while (pos.y < H) {
      // Find a position that is not occupied by a node.
      const fpos = reactFlow.screenToFlowPosition(pos);
      const occupied = crdt?.ws?.nodes?.some((n) => {
        const np = n.position;
        return (
          np.x < fpos.x + w + GAP &&
          np.x + (n.width ?? 0) + GAP > fpos.x &&
          np.y < fpos.y + h + GAP &&
          np.y + (n.height ?? 0) + GAP > fpos.y
        );
      });
      if (!occupied) {
        return pos;
      }
      // Move the position to the right and down until we find a free spot.
      pos.x += SPEED;
      if (pos.x + w > W) {
        pos.x = 100;
        pos.y += SPEED;
      }
    }
    return { x: 100, y: 100 };
  }

  function isTypingInFormElement() {
    const activeElement = document.activeElement;
    return (
      activeElement &&
      (activeElement.tagName === "INPUT" ||
        activeElement.tagName === "TEXTAREA" ||
        (activeElement as HTMLElement).isContentEditable)
    );
  }

  const closeNodeSearch = useCallback(() => {
    setNodeSearchSettings(undefined);
    setSuppressSearchUntil(Date.now() + 200);
  }, []);
  const toggleNodeSearch = useCallback(
    (event: MouseEvent) => {
      if (!canWrite) return;
      if (!categoryHierarchy) return;
      if (suppressSearchUntil > Date.now()) return;
      if (nodeSearchSettings) {
        closeNodeSearch();
        return;
      }
      event.preventDefault();
      setNodeSearchSettings({
        pos: { x: event.clientX, y: event.clientY },
      });
    },
    [categoryHierarchy, canWrite, nodeSearchSettings, suppressSearchUntil, closeNodeSearch],
  );
  function findFreeId(prefix: string) {
    let i = 1;
    let id = `${prefix} ${i}`;
    const used = new Set(crdt?.ws?.nodes?.map((n) => n.id));
    while (used.has(id)) {
      i += 1;
      id = `${prefix} ${i}`;
    }
    return id;
  }
  function addNode(node: Partial<WorkspaceNode>) {
    crdt?.addNode(node);
  }
  function nodeFromMeta(meta: OpsOp): Partial<WorkspaceNode> {
    const node: Partial<WorkspaceNode> = {
      type: meta.type,
      width: 315,
      height: 315,
      data: {
        meta: meta,
        title: meta.name,
        op_id: meta.id || meta.name,
        params: Object.fromEntries(meta.params.map((p) => [p.name, p.default])),
      },
    };
    return node;
  }
  const addNodeFromSearch = useCallback(
    (meta: OpsOp) => {
      const node = nodeFromMeta(meta);
      const nss = nodeSearchSettings!;
      node.position = reactFlow.screenToFlowPosition({
        x: nss.pos.x,
        y: nss.pos.y,
      });
      node.id = findFreeId(node.data!.title);
      addNode(node);
      closeNodeSearch();
    },
    [nodeSearchSettings, reactFlow, closeNodeSearch],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setSuppressSearchUntil(Date.now() + 200);
      const edge = {
        id: `${connection.source} ${connection.sourceHandle} ${connection.target} ${connection.targetHandle}`,
        source: connection.source,
        sourceHandle: connection.sourceHandle!,
        target: connection.target,
        targetHandle: connection.targetHandle!,
      };
      crdt?.addEdge(edge);
    },
    [crdt],
  );
  const parentDir = parentPath(path!);
  function onDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.stopPropagation();
    e.preventDefault();
  }
  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    cursorScreenPos.current = { x: e.clientX, y: e.clientY };
  }
  async function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.stopPropagation();
    e.preventDefault();
    if (!canWrite) return;
    const file = e.dataTransfer.files[0];
    if (!catalog.data || !crdt?.ws?.env) {
      return;
    }
    try {
      await uploadFile(file, {
        onProgress: (percentCompleted) => {
          if (percentCompleted === 100) setMessage("Processing file...");
          else setMessage(`Uploading ${percentCompleted}%`);
        },
      });
      setMessage(null);
      const cat = catalog.data[crdt.ws.env];
      const node = nodeFromMeta(cat["Import file"]);
      node.id = findFreeId(node.data!.title);
      node.position = reactFlow.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });
      node.data!.params.file_path = `uploads/${file.name}`;
      if (file.name.includes(".csv")) {
        node.data!.params.file_format = "csv";
      } else if (file.name.includes(".parquet")) {
        node.data!.params.file_format = "parquet";
      } else if (file.name.includes(".json")) {
        node.data!.params.file_format = "json";
      } else if (file.name.includes(".xls")) {
        node.data!.params.file_format = "excel";
      } else if (file.name.includes(".cif")) {
        node.data!.params.file_format = "cif";
      }
      addNode(node);
    } catch (error) {
      setMessage("File upload failed.");
      console.error("File upload failed.", error);
    }
  }
  async function executeWorkspace() {
    const response = await axios.post(`/api/execute_workspace?name=${encodeURIComponent(path)}`);
    if (response.status !== 200) {
      setMessage("Workspace execution failed.");
    }
  }
  const selectAll = useCallback(() => {
    if (crdt?.onFENodesChange) {
      crdt.onFENodesChange(nodes.map((n) => ({ id: n.id, type: "select", selected: true })));
    }
    if (crdt?.onFEEdgesChange) {
      crdt.onFEEdgesChange(edges.map((e) => ({ id: e.id, type: "select", selected: true })));
    }
  }, [crdt, nodes, edges]);
  function deleteSelection() {
    const selectedNodeIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id));
    const edgesToRemoveIds = new Set(
      edges
        .filter(
          (edge) =>
            edge.selected || selectedNodeIds.has(edge.source) || selectedNodeIds.has(edge.target),
        )
        .map((e) => e.id),
    );
    if (selectedNodeIds.size === 0 && edgesToRemoveIds.size === 0) {
      return;
    }
    crdt?.applyChange((conn) => {
      const wnodes = conn.ws.get("nodes") as YArray<any>;
      const nodeIndices: number[] = [];
      let nodeIdx = 0;
      for (const node of wnodes) {
        if (selectedNodeIds.has((node as YMap<any>).get("id") as string)) {
          nodeIndices.push(nodeIdx);
        }
        nodeIdx += 1;
      }
      for (const idx of nodeIndices.sort((a, b) => b - a)) {
        wnodes.delete(idx);
      }
      const wedges = conn.ws.get("edges") as YArray<any>;
      const edgeIndices: number[] = [];
      let edgeIdx = 0;
      for (const edge of wedges) {
        if (edgesToRemoveIds.has((edge as YMap<any>).get("id") as string)) {
          edgeIndices.push(edgeIdx);
        }
        edgeIdx += 1;
      }
      for (const idx of edgeIndices.sort((a, b) => b - a)) {
        wedges.delete(idx);
      }
    });
  }
  function changeBox() {
    const [selectedNode] = nodes.filter((n) => n.selected);
    reactFlow.updateNodeData(selectedNode.id, { op_id: "" });
  }
  function groupSelection() {
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    const groupNode = {
      id: findFreeId("Group"),
      type: "node_group",
      position: { x: 0, y: 0 },
      width: 0,
      height: 0,
      data: { title: "Group", params: {} },
    };
    let top = Number.POSITIVE_INFINITY;
    let left = Number.POSITIVE_INFINITY;
    let bottom = Number.NEGATIVE_INFINITY;
    let right = Number.NEGATIVE_INFINITY;
    const PAD = 50;
    for (const node of selectedNodes) {
      if (node.position.y - PAD < top) top = node.position.y - PAD;
      if (node.position.x - PAD < left) left = node.position.x - PAD;
      if (node.position.y + PAD + node.height! > bottom)
        bottom = node.position.y + PAD + node.height!;
      if (node.position.x + PAD + node.width! > right) right = node.position.x + PAD + node.width!;
      node.selected = false;
    }
    groupNode.position = {
      x: left,
      y: top,
    };
    groupNode.width = right - left;
    groupNode.height = bottom - top;
    crdt.applyChange((conn) => {
      const wnodes = conn.ws.get("nodes") as YArray<any>;
      wnodes.unshift([nodeToYMap(groupNode)]);
      const selectedNodesById = new Map(selectedNodes.map((n) => [n.id, n]));
      for (const node of wnodes) {
        const feNode = selectedNodesById.get(node.get("id"));
        if (feNode) {
          const pos = feNode.position;
          node.set("position", {
            x: pos.x - left,
            y: pos.y - top,
          });
          node.set("parentId", groupNode.id);
          node.set("extent", "parent");
        }
      }
    });
    crdt.onFENodesChange?.([
      ...selectedNodes.map((n) => ({ id: n.id, type: "select" as const, selected: false })),
      { id: groupNode.id, type: "select" as const, selected: true },
    ]);
  }
  function ungroupSelection() {
    const groups = Object.fromEntries(
      nodes
        .filter((n) => n.selected && n.type === "node_group" && !n.parentId)
        .map((n) => [n.id, n]),
    );
    const childNodeIds = nodes.filter((n) => n.parentId && n.parentId in groups).map((n) => n.id);
    crdt.applyChange((conn) => {
      const wnodes = conn.ws.get("nodes") as YArray<YMap<any>>;
      for (const node of wnodes) {
        const g = groups[node.get("parentId") as string];
        if (!g) continue;
        const posValue = node.get("position");
        // Yjs position values can be either a plain object or a Y.Map with a toJSON method.
        const pos = (posValue.toJSON ? posValue.toJSON() : posValue) as XYPosition;
        node.set("position", {
          x: pos.x + g.position.x,
          y: pos.y + g.position.y,
        });
        node.delete("parentId");
        node.delete("extent");
      }
      const groupIndices: number[] = wnodes
        .map((n: any, idx: number) => ({ id: n.get("id"), idx }))
        .filter(({ id }: { id: string }) => id in groups)
        .map(({ idx }: { idx: number }) => idx);
      groupIndices.sort((a, b) => b - a);
      for (const groupIdx of groupIndices) {
        wnodes.delete(groupIdx, 1);
      }
    });
    crdt.onFENodesChange?.(
      childNodeIds.map((id) => ({ id, type: "select" as const, selected: true })),
    );
  }
  const selected = nodes.filter((n) => n.selected);
  const isAnyGroupSelected = nodes.some((n) => n.selected && n.type === "node_group");

  if (!permissions.isLoading && !permissions.read) {
    return (
      <div className="workspace">
        <div className="hero min-h-screen">
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title">No access</h2>
              <p>You do not have permission to view this workspace.</p>
              <div className="card-actions justify-end">
                <Link to="/" className="btn btn-primary">
                  Back
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`workspace${canWrite ? "" : " read-only"}`}>
      <div className="top-bar bg-neutral">
        <div className="top-bar-leading">
          <Link className="logo" to="/">
            <img alt="" src={favicon} />
          </Link>
          <div className="ws-name">{shortPath}</div>
          {!canWrite && !permissions.isLoading && (
            <span className="badge badge-ghost ml-2">Read-only</span>
          )}
        </div>
        <title>{shortPath}</title>
        <div className="top-bar-trailing">
          <WorkspaceProgress path={path} enabled={Boolean(crdt?.ws)} />
          {crdt?.ws && canWrite && (
            <div className="top-bar-controls">
              <ExecutionOptions
                env={crdt.ws.env || ""}
                value={crdt.ws.execution_options}
                onChange={crdt.setExecutionOptions}
              />
              <EnvironmentSelector
                options={Object.keys(catalog.data || {})}
                value={crdt.ws.env || ""}
                onChange={crdt.setEnv}
              />
            </div>
          )}
          <div className="tools text-secondary">
            {crdt?.ws && canWrite && (
              <>
                <Tooltip doc="Group selected nodes">
                  <button
                    className="btn btn-link"
                    disabled={selected.length < 2}
                    onClick={groupSelection}
                    name="groupBtn"
                  >
                    <GroupIcon />
                  </button>
                </Tooltip>
                <Tooltip doc="Ungroup selected nodes">
                  <button
                    className="btn btn-link"
                    disabled={!isAnyGroupSelected}
                    onClick={ungroupSelection}
                    name="ungroupBtn"
                  >
                    <UngroupIcon />
                  </button>
                </Tooltip>
                <Tooltip doc="Delete selected nodes and edges">
                  <button
                    className="btn btn-link"
                    disabled={selected.length === 0}
                    onClick={deleteSelection}
                  >
                    <DeleteIcon />
                  </button>
                </Tooltip>
                <Tooltip doc="Change selected box to a different box">
                  <button
                    className="btn btn-link"
                    disabled={selected.length !== 1}
                    onClick={changeBox}
                  >
                    <ChangeTypeIcon />
                  </button>
                </Tooltip>
                <Tooltip doc={gridSnapEnabled ? "Disable grid snapping" : "Enable grid snapping"}>
                  <button
                    className="btn btn-link"
                    onClick={() => setGridSnapEnabled(!gridSnapEnabled)}
                  >
                    {gridSnapEnabled ? <GridIcon /> : <GridOffIcon />}
                  </button>
                </Tooltip>
                {config.assistant_available && (
                  <Tooltip doc={"Toggle assistant"}>
                    <button
                      className="btn btn-link"
                      onClick={() => setIsAssistantOpen(!isAssistantOpen)}
                    >
                      <RobotIcon />
                    </button>
                  </Tooltip>
                )}
                <Tooltip
                  doc={crdt.ws.paused ? "Resume automatic execution" : "Pause automatic execution"}
                >
                  <button
                    className="btn btn-link"
                    onClick={() => crdt.setPausedState(!crdt.ws?.paused)}
                  >
                    {crdt.ws.paused ? <PlayIcon /> : <PauseIcon />}
                  </button>
                </Tooltip>
                <Tooltip doc="Re-run the workspace">
                  <button className="btn btn-link" onClick={executeWorkspace}>
                    <RestartIcon />
                  </button>
                </Tooltip>
              </>
            )}
            <Tooltip doc="Close workspace">
              <Link
                className="btn btn-link"
                to={`/dir/${parentDir
                  .split("/")
                  .map((segment) => encodeURIComponent(segment))
                  .join("/")}`}
                aria-label="close"
              >
                <CloseIcon />
              </Link>
            </Tooltip>
            <UserMenu />
          </div>
        </div>
      </div>
      <div className="workspace-body">
        <div
          className="reactflow-container"
          onDragOver={canWrite ? onDragOver : undefined}
          onMouseMove={onMouseMove}
          onDrop={canWrite ? onDrop : undefined}
          ref={reactFlowContainer}
        >
          {crdt?.ws ? (
            <LynxKiteState.Provider value={{ workspace: crdt.ws, iconized, canWrite }}>
              <ReactFlow
                nodes={nodes}
                edges={autoConnect.renderedEdges}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                nodesDraggable={canWrite}
                nodesConnectable={canWrite}
                elementsSelectable={true}
                deleteKeyCode={canWrite ? ["Backspace", "Delete"] : null}
                onNodesChange={(changes) => {
                  changes = snapChangesToGrid(
                    changes,
                    isShiftPressed || gridSnapEnabled,
                    crdt?.ws?.nodes || [],
                  );
                  crdt?.onFENodesChange?.(changes);
                }}
                onEdgesChange={crdt?.onFEEdgesChange}
                onPaneClick={canWrite ? toggleNodeSearch : undefined}
                onConnect={canWrite ? onConnect : undefined}
                onNodeDrag={canWrite ? autoConnect.onNodeDrag : undefined}
                onNodeDragStop={canWrite ? autoConnect.onNodeDragStop : undefined}
                onMove={() => {
                  const zoom = reactFlow.getZoom();
                  setIconized(zoom < ICONIZE_THRESHOLD);
                }}
                proOptions={{ hideAttribution: true }}
                maxZoom={10}
                minZoom={0.1}
                zoomOnScroll={true}
                panOnScroll={false}
                panOnDrag={[0]}
                selectionOnDrag={false}
                preventScrolling={true}
                defaultEdgeOptions={{
                  markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: "#888",
                    width: 15,
                    height: 15,
                  },
                }}
                fitViewOptions={{ maxZoom: 1 }}
              >
                <Background
                  variant={BackgroundVariant.Dots}
                  gap={35}
                  size={6}
                  color="#f0f0f0"
                  bgColor="#fafafa"
                  offset={3}
                />
                {nodeSearchSettings && categoryHierarchy && canWrite && (
                  <NodeSearch
                    pos={nodeSearchSettings.pos}
                    categoryHierarchy={categoryHierarchy}
                    onCancel={closeNodeSearch}
                    onClick={addNodeFromSearch}
                  />
                )}
              </ReactFlow>
            </LynxKiteState.Provider>
          ) : (
            <div className="workspace-loading">Loading workspace...</div>
          )}
          {message && (
            <div className="workspace-message">
              <span className="close" onClick={() => setMessage(null)}>
                <Close />
              </span>
              {message}
            </div>
          )}
        </div>
        {isAssistantOpen && canWrite && (
          <Suspense fallback={<aside className="assistant-panel" />}>
            <Assistant crdtWorkspace={crdt} />
          </Suspense>
        )}
      </div>
    </div>
  );
}
