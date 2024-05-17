import App from './App.svelte';

import './app.scss';
import * as bootstrap from 'bootstrap';

const app = new App({
  target: document.getElementById('app')!,
});

export default app;
