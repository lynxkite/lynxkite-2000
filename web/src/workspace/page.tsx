'use client';
import { useMemo } from "react";
import dynamic from 'next/dynamic';

export default dynamic(() => import('./Workspace'), {
  ssr: false,
});
