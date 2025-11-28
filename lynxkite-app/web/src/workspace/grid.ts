// Grid snapping functions.

// Snaps XYFlow changes to the grid if Shift is pressed.
export function snapChangesToGrid(changes: any[], isShiftPressed: boolean, nodes: any[]) {
  // Grid size for snapping
  const GRID_SIZE = 40;

  function snapToGrid(position: { x: number; y: number }) {
    return {
      x: Math.round(position.x / GRID_SIZE) * GRID_SIZE,
      y: Math.round(position.y / GRID_SIZE) * GRID_SIZE,
    };
  }

  function snapDimensionsToGrid(
    dimensions: { width: number; height: number },
    nodePosition: { x: number; y: number },
  ) {
    // Calculate where the bottom-right corner should be
    const rightEdge = nodePosition.x + dimensions.width;
    const bottomEdge = nodePosition.y + dimensions.height;

    // Snap the bottom-right corner to grid
    const snappedRightEdge = Math.round(rightEdge / GRID_SIZE) * GRID_SIZE;
    const snappedBottomEdge = Math.round(bottomEdge / GRID_SIZE) * GRID_SIZE;

    // Calculate new dimensions based on snapped edges
    return {
      width: Math.max(GRID_SIZE, snappedRightEdge - nodePosition.x),
      height: Math.max(GRID_SIZE, snappedBottomEdge - nodePosition.y),
    };
  }

  return changes.map((ch) => {
    if (
      ch.type === "position" &&
      !Number.isNaN(ch.position.x) &&
      !Number.isNaN(ch.position.y) &&
      isShiftPressed
    ) {
      // Snap to grid when Shift is pressed
      return {
        ...ch,
        position: snapToGrid(ch.position),
      };
    } else if (
      ch.type === "dimensions" &&
      ch.dimensions &&
      !Number.isNaN(ch.dimensions.width) &&
      !Number.isNaN(ch.dimensions.height) &&
      isShiftPressed
    ) {
      // Find the node to get its position
      const node = nodes.find((n) => n.id === ch.id);
      if (node) {
        // Snap dimensions to grid when Shift is pressed
        return {
          ...ch,
          dimensions: snapDimensionsToGrid(ch.dimensions, node.position),
        };
      }
    }
    return ch;
  });
}
