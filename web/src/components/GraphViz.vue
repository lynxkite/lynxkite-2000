<template>
    <div ref="container" class="graph-viz"></div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import Sigma from 'sigma'
import * as graphology from 'graphology'
const container = ref(null)
onMounted(() => {
  const sigmaInstance = new Sigma(graph, container.value);
})
defineProps(['modelValue'])
// Create a graphology graph
const graph = new graphology.Graph();
graph.addNode("1", { label: "Node 1", x: 0, y: 0, size: 10, color: "blue" });
graph.addNode("2", { label: "Node 2", x: 1, y: 1, size: 20, color: "red" });
graph.addEdge("1", "2", { size: 5, color: "purple" });

// Instantiate sigma.js and render the graph

</script>
