import LynxKiteNode from "./LynxKiteNode";
import { NodeWithParams } from "./NodeWithParams";

const NodeWithImage = (props: any) => {
  return (
    <NodeWithParams collapsed {...props}>
      {props.data.display && <img src={props.data.display} alt="Node Display" />}
    </NodeWithParams>
  );
};

export default LynxKiteNode(NodeWithImage);
