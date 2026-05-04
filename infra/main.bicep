@description('Environment name: dev or prod')
param environmentName string

@description('Azure region for all resources')
param location string = resourceGroup().location

// ── Naming ───────────────────────────────────────────────────────────────────
var suffix = uniqueString(resourceGroup().id)
var storageAccountName    = 'amlworkshop${environmentName}st${substring(suffix, 0, 8)}'
var keyVaultName          = 'kv-${environmentName}-${substring(suffix, 0, 10)}'
var appInsightsName       = 'amlworkshop-${environmentName}-appi'
var containerRegistryName = 'amlworkshop${environmentName}cr${substring(suffix, 0, 8)}'
var workspaceName         = 'amlworkshop-${environmentName}-workspace'

// ── Storage ───────────────────────────────────────────────────────────────────
resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
  }
}

// ── Key Vault ─────────────────────────────────────────────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: tenant().tenantId
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enableRbacAuthorization: false   // AML manages its own access policies internally
    accessPolicies: []               // AML appends its own policy at workspace creation
  }
}

// ── Application Insights ──────────────────────────────────────────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    RetentionInDays: 30
  }
}

// ── Container Registry ────────────────────────────────────────────────────────
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true
  }
}

// ── Azure ML Workspace ────────────────────────────────────────────────────────
resource workspace 'Microsoft.MachineLearningServices/workspaces@2023-10-01' = {
  name: workspaceName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount:       storage.id
    keyVault:             keyVault.id
    applicationInsights:  appInsights.id
    containerRegistry:    containerRegistry.id
    publicNetworkAccess:  'Enabled'
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────
output workspaceName        string = workspace.name
output storageAccountName   string = storage.name
output workspacePrincipalId string = workspace.identity.principalId
output storageId            string = storage.id
output acrId                string = containerRegistry.id
