# Renderiza todos os videos da demo + FOTOS dos momentos com mais acontecimentos.
# Uso (na raiz do projeto):  powershell -ExecutionPolicy Bypass -File scripts\render_demo.ps1
# Saida: runs\VIDEOS_DEMO\1_finais\*.mp4, 2_versoes_antigas\*.mp4 e runs\VIDEOS_DEMO\fotos\<video>\*.png
# (comentarios sem acento: o PowerShell 5.1 le .ps1 sem BOM como ANSI)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore'
$final = 'models\detector_final.pt'
# nome | pasta destino | video | cena | roteiro | pesos | n fotos | intervalo (s) | args extras
$runs = @(
  @('01_avenida_vista_de_cima_entrou_sem_seta', '1_finais', 'data\videos\demo_avenida_cruzamento.mp4', 'data\scenes\pexels_avenida.json', 'data\roteiros\demo_avenida_cruzamento.json', $final, 8, 1.5, @('--light', 'unknown', '--min-box', '0.02')),
  @('02_cruzamento_batida_fuga_e_seta', '1_finais', 'data\videos\uadetrac\MVI_40701.mp4', 'data\scenes\uadetrac_cruzamento.json', 'data\roteiros\MVI_40701.json', $final, 10, 1.5, @('--light', 'unknown')),
  @('03_avancou_vermelho_e_bateu', '1_finais', 'data\videos\tudat_v36_sinal_vermelho.mp4', 'data\scenes\tudat_v36.json', 'data\roteiros\tudat_v36.json', $final, 6, 0.6, @('--light', 'auto')),
  @('04_avancou_vermelho_e_bateu_camera_fiscalizacao', '1_finais', 'data\videos\tudat_v40_sinal_vermelho.mp4', 'data\scenes\tudat_v40.json', 'data\roteiros\tudat_v40.json', $final, 6, 0.6, @('--light', 'auto')),
  @('05_tunel_bateu_e_fugiu', '1_finais', 'data\videos\accident_tunel_fuga.mp4', 'data\scenes\accident_tunel.json', 'data\roteiros\accident_tunel_fuga.json', $final, 6, 0.8, @('--light', 'unknown', '--min-box', '0.03', '--conf', '0.12')),
  @('06_EPTV_av_saudade_derrubou_moto_e_fugiu', '1_finais', 'data\videos\eptv_saudade.mp4', 'data\scenes\eptv_saudade.json', 'data\roteiros\eptv_saudade.json', $final, 6, 0.6, @('--light', 'unknown', '--min-box', '0.04', '--conf', '0.25')),
  @('cruzamento_roteiro_modelo_visdrone', '2_versoes_antigas', 'data\videos\uadetrac\MVI_40701.mp4', 'data\scenes\uadetrac_cruzamento.json', 'data\roteiros\MVI_40701.json', 'runs\train\visdrone_vehicles\weights\best.pt', 10, 1.5, @('--light', 'unknown')),
  @('avenida_roteiro_modelo_parcial', '2_versoes_antigas', 'data\videos\demo_avenida_cruzamento.mp4', 'data\scenes\pexels_avenida.json', 'data\roteiros\demo_avenida_cruzamento.json', 'models\uadetrac_ft_parcial.pt', 8, 1.5, @('--light', 'unknown', '--min-box', '0.02'))
)
foreach ($r in $runs) {
  $name, $dest, $video, $scene, $rot, $w, $n, $gap, $extra = $r
  $out = "runs\demo\render_$name"
  $fotos = "runs\VIDEOS_DEMO\fotos\$name"
  if (Test-Path $fotos) { Remove-Item $fotos -Recurse -Force }
  $t0 = Get-Date
  & .\.venv\Scripts\python.exe scripts\run_pipeline.py --video $video --scene $scene --roteiro $rot --weights $w --out $out `
      --frames-out $fotos --frames-n $n --frames-gap $gap @extra 2>&1 | Out-Null
  Copy-Item "$out\video_anotado.mp4" "runs\VIDEOS_DEMO\$dest\$name.mp4" -Force
  $inf = (Get-Content "$out\infracoes.jsonl" | ForEach-Object { $j = $_ | ConvertFrom-Json; "$($j.type)#$($j.track_id)[$($j.plate)]" }) -join ', '
  "{0}: {1:N0}s | {2} fotos | {3}" -f $name, ((Get-Date) - $t0).TotalSeconds, (Get-ChildItem $fotos -ErrorAction SilentlyContinue).Count, $inf
}
