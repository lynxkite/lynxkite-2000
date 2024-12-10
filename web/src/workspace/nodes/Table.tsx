export default function Table(props: any) {
  return (<table>
    <thead>
      <tr>
        {props.columns.map((column: string) =>
          <th key={column}>{column}</th>)}
      </tr>
    </thead>
    <tbody>
      {props.data.map((row: { [column: string]: any }, i: number) =>
        <tr key={`row-${i}`}>
          {props.columns.map((_column: string, j: number) =>
            <td key={`cell ${i}, ${j}`}>{JSON.stringify(row[j])}</td>)}
        </tr>)}
    </tbody>
  </table>);
}
