// Baklava feature requests:
// - Node search (https://github.com/newcat/baklavajs/issues/315)
// - Check out sidebar
// - Collapse node
// - Customize toolbar
// - Nicer styling for node palette, node categories
// - Group nodes, like in Litegraph.
// - Show param name for TextInputInterface.
// - Resizable nodes. (Horizontally.)
// - Put input and output on same line?

<template>
  <div style="width: 100vw; height: 100vh">
    <BaklavaEditor :view-model="baklava" />
  </div>
</template>

<script setup lang="ts">
import { BaklavaEditor, useBaklava } from "@baklavajs/renderer-vue";
import * as BaklavaJS from "baklavajs";
import "@baklavajs/themes/dist/syrup-dark.css";
import { markRaw } from "vue";
import { NodeInterface } from "baklavajs";
import GraphViz from "./components/GraphViz.vue";

const ops = [
  { name: 'Create scale-free graph', type: 'Creation', inputs: [], outputs: ['graph'], params: ['Number of nodes'] },
  { name: 'Compute PageRank', type: 'Algorithms', inputs: ['graph'], outputs: ['graph'], params: ['Damping factor', 'Max iterations'] },
  { name: 'SQL', type: 'Algorithms', inputs: ['graph'], outputs: ['table'], params: ['Query'] },
  { name: 'Visualize graph', type: 'Visualization', inputs: ['graph'], outputs: ['graph-viz'],
    params: ['Color by', 'Size by'],
    calculate(inputs) {
      console.log('Visualize graph', inputs);
      return {
        'graph-viz': 15,
      };
    }
  },
  { name: 'Num', type: 'Visualization', inputs: [], outputs: ['num'], params: [] },
];

function makeParam(param: string): NodeInterface {
  return new BaklavaJS.TextInputInterface(param).setPort(false);
}

function makeOutput(output: string): NodeInterface {
  if (output === 'graph-viz') {
    return new NodeInterface(output, 0).setComponent(markRaw(GraphViz)).setPort(false);
  } else if (output === 'num') {
    return new BaklavaJS.NumberInterface(output, 0);
  } else {
    return new BaklavaJS.NodeInterface(output, 0);
  }
}

const baklava = useBaklava();
for (const op of ops) {
  baklava.editor.registerNodeType(BaklavaJS.defineNode({
    type: op.name,
    inputs: {
      ...op.inputs.reduce((acc, input) => ({ ...acc, [input]: () => new BaklavaJS.NodeInterface(input) }), {}),
      ...op.params.reduce((acc, param) => ({ ...acc, [param]: () => makeParam(param)}), {}),
    },
    outputs: op.outputs.reduce((acc, output) => ({ ...acc, [output]: () => makeOutput(output) }), {}),
    calculate: op.calculate,
  }), { category: op.type });
}

import { DependencyEngine } from "@baklavajs/engine";
// Needed?
const engine = new DependencyEngine(baklava.editor);
engine.start();

import { applyResult } from "@baklavajs/engine";
// Needed?
const token = Symbol();
engine.events.afterRun.subscribe(token, (result) => {
    engine.pause();
    applyResult(result, baklava.editor);
    engine.resume();
});
</script>
