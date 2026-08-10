import { useEffect, useState } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import { attachWorkspaceEtaAnchor, parseProgressWorkspace } from "../progress";

export function useWorkspaceProgress(roomName: string | undefined, enabled = true) {
  const [workspaceProgress, setWorkspaceProgress] = useState<any | null>(null);

  useEffect(() => {
    if (!enabled || !roomName) {
      setWorkspaceProgress(null);
      return;
    }
    const currentRoomName = roomName;

    const doc = new Y.Doc();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const provider = new WebsocketProvider(
      `${proto}//${location.host}/ws/progress/crdt`,
      "progress",
      doc,
    );
    const wsMap = doc.getMap("workspaces");

    function syncWorkspaceProgress() {
      const parsed = parseProgressWorkspace(wsMap.get(currentRoomName));
      setWorkspaceProgress((prev: any) => attachWorkspaceEtaAnchor(parsed, prev));
    }

    wsMap.observe(syncWorkspaceProgress);
    provider.on("sync", syncWorkspaceProgress);
    provider.on("status", (event: any) => {
      if (event?.status === "connected") {
        syncWorkspaceProgress();
      }
    });

    return () => {
      wsMap.unobserve(syncWorkspaceProgress);
      provider.destroy();
      doc.destroy();
    };
  }, [enabled, roomName]);

  return workspaceProgress;
}
