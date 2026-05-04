@description('Environment name: dev or prod')
param environmentName string

@description('Azure region for all resources')
param location string = resourceGroup().location

// ── Naming ───────────────────────────────────────────────────────────────────
var suffix = uniqueString(resourceGroup().id)
var storageAccountName    = 'aml${environmentName}st${substring(suffix, 0, 8)}'
var keyVaultName          = 'aml-${environmentName}-kv-${substring(suffix, 0, 6)}'
var appInsightsName       = 'aml-${environmentName}-appi'
var containerRegistryName = 'aml${environmentName}cr${substring(suffix, 0, 8)}'
var workspaceName         = 'aml-${environmentName}-workspace'

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
    enableRbacAuthorization: true
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
    adminUserEnabled: false
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

// ── Role assignments for workspace managed identity ───────────────────────────
// Storage Blob Data Contributor — read/write job artifacts
resource storageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, workspace.id, 'StorageBlobDataContributor')
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
    )
    principalId:   workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// AcrPush — build and push custom environment images
resource acrRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, workspace.id, 'AcrPush')
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '8311e382-0749-4cb8-b61a-304f252e45ec'
    )
    principalId:   workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Key Vault Secrets Officer — store connections and credentials
resource kvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, workspace.id, 'KeyVaultSecretsOfficer')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
    )
    principalId:   workspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Outputs ───────────────────────────────────────────────────────────────────
output workspaceName     string = workspace.name
output storageAccountName string = storage.name
