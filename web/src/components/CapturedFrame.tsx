interface Props {
  src: string;
  tag: string;
  scanning: boolean;
}

export function CapturedFrame({ src, tag, scanning }: Props) {
  return (
    <div className="frame">
      <img src={src} alt="Загруженный кадр" />
      <span className="frame-tag">{tag}</span>
      {scanning && <div className="frame-scan" aria-hidden="true" />}
    </div>
  );
}
