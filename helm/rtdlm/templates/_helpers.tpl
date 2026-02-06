{{/*
Expand the name of the chart.
*/}}
{{- define "rtdlm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "rtdlm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "rtdlm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "rtdlm.labels" -}}
helm.sh/chart: {{ include "rtdlm.chart" . }}
{{ include "rtdlm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "rtdlm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rtdlm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Training labels
*/}}
{{- define "rtdlm.training.labels" -}}
{{ include "rtdlm.labels" . }}
app.kubernetes.io/component: training
{{- end }}

{{/*
Training selector labels
*/}}
{{- define "rtdlm.training.selectorLabels" -}}
{{ include "rtdlm.selectorLabels" . }}
app.kubernetes.io/component: training
{{- end }}

{{/*
Metrics labels
*/}}
{{- define "rtdlm.metrics.labels" -}}
{{ include "rtdlm.labels" . }}
app.kubernetes.io/component: metrics
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "rtdlm.serviceAccountName" -}}
{{- if .Values.training.serviceaccount }}
{{- .Values.training.serviceaccount }}
{{- else }}
{{- include "rtdlm.fullname" . }}-sa
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "rtdlm.imagePullSecrets" -}}
{{- if .Values.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Training image
*/}}
{{- define "rtdlm.training.image" -}}
{{- printf "%s:%s" .Values.training.image.repository .Values.training.image.tag }}
{{- end }}

{{/*
Full DNS name for service
*/}}
{{- define "rtdlm.serviceDNS" -}}
{{- printf "%s.%s.svc.cluster.local" .Values.training.service.name .Values.Namespace }}
{{- end }}

{{/*
Ingress hostname
*/}}
{{- define "rtdlm.ingressHost" -}}
{{- printf "%s.%s.%s" .Values.training.ingress.subdomain .Values.dns.subdomain .Values.dns.basedomain }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "rtdlm.commonEnv" -}}
- name: PYTHONUNBUFFERED
  value: "1"
- name: PYTHONDONTWRITEBYTECODE
  value: "1"
- name: XLA_PYTHON_CLIENT_PREALLOCATE
  value: {{ .Values.training.jax.preallocate | quote }}
- name: XLA_PYTHON_CLIENT_MEM_FRACTION
  value: {{ .Values.training.jax.mem_fraction | quote }}
{{- end }}

{{/*
Model environment variables
*/}}
{{- define "rtdlm.modelEnv" -}}
- name: MODEL_PRESET
  valueFrom:
    configMapKeyRef:
      name: {{ include "rtdlm.fullname" . }}-config
      key: MODEL_PRESET
- name: BATCH_SIZE
  valueFrom:
    configMapKeyRef:
      name: {{ include "rtdlm.fullname" . }}-config
      key: BATCH_SIZE
- name: LEARNING_RATE
  valueFrom:
    configMapKeyRef:
      name: {{ include "rtdlm.fullname" . }}-config
      key: LEARNING_RATE
- name: EPOCHS
  valueFrom:
    configMapKeyRef:
      name: {{ include "rtdlm.fullname" . }}-config
      key: EPOCHS
- name: CHECKPOINT_DIR
  value: "/checkpoints"
- name: LOG_DIR
  value: "/logs"
- name: DATA_DIR
  value: {{ .Values.training.data.path | quote }}
{{- end }}

{{/*
Observability labels for Datadog/Prometheus
*/}}
{{- define "rtdlm.observabilityLabels" -}}
tags.datadoghq.com/service: {{ .Values.observability_tags.training.servicename }}
tags.datadoghq.com/version: {{ .Values.observability_tags.product.version }}
tags.datadoghq.com/env: {{ .Values.observability_tags.env }}
tags.datadoghq.com/env_name: {{ .Values.observability_tags.env_name }}
{{- end }}

{{/*
Prometheus annotations
*/}}
{{- define "rtdlm.prometheusAnnotations" -}}
{{- if .Values.training.metrics.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .Values.training.metrics.port | quote }}
prometheus.io/path: {{ .Values.training.metrics.path | quote }}
{{- end }}
{{- end }}
