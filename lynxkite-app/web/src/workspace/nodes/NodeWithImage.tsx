import { useDisplay } from "../../common.ts";
import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

const NodeWithImage = (props: any) => {
  const display = useDisplay(props.data?.display_version, props.id);
  return (
    <NodeWithParams collapsed {...props}>
      {display && <img src={display} alt="Node Display" />}
    </NodeWithParams>
  );
};

export default LynxKiteNode(NodeWithImage);
