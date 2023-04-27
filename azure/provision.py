# Requires the Azure CLI to run
# Can be installed on macOS using
# brew update && brew install azure-cli
# See: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

import os, time, re, random

from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient

from fabric import Connection
from paramiko.ssh_exception import NoValidConnectionsError

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives import serialization

class RandomName:
  name = None
  def __init__(self, prefix, length):
    generated = '{{:0{}x}}'.format(length).format(random.randint(0, 16**length))
    search = re.compile('[ \t!@£$%^&*(){}\[\],/<>?`;\':"]')
    name = search.sub('-', prefix)
    self.name = '{}-{}'.format(name, generated)

  def get(self):
    return self.name

class VirtualMachine:
  ip_address = None
  comute_client = None
  location = 'uksouth'
  subscription_id = None

  # Azure nameing options
  prefix = None
  resource_group_name = None
  vnet_name = None
  subnet_name = None
  ip_name = None
  ip_config_name = None
  nic_name = None
  nsg_name = None
  vm_name = None

  # Authentication options
  ssh_key_file = None
  username = 'azure'
  ssh_public_public = None

  def __init__(self, prefix, ssh_key_file, subscription_id):
    self.set_prefix(prefix)
    self.ssh_key_file = ssh_key_file
    self._load_public_key()
    self.subscription_id = subscription_id

  def _check_result(self, title, result):
    if result.id:
      print('Created {}'.format(title))
      print('Name: {}'.format(result.name))
      print()
    else:
      print('Creation of {} failed'.format(title))
      print('Output: {}'.format(result))
      exit()

  def _load_public_key(self):
    with open(self.ssh_key_file, 'rb') as file:
      data = file.read()
      key = load_pem_private_key(data, password=None)
      # Decode the public key
      # Ensure we add a user to the end or it won't be accepted
      self.ssh_public_key = key.public_key().public_bytes(encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH).decode('utf-8')

  def set_prefix(self, prefix):
    search = re.compile('[ \t!@£$%^&*(){}\[\],/<>?`;\':"]')
    self.prefix = search.sub('-', prefix)
    self.resource_group_name = f'{self.prefix}-rg'
    self.vnet_name = f'{self.prefix}-vnet'
    self.subnet_name = f'{self.prefix}-subnet'
    self.ip_name = f'{self.prefix}-ip'
    self.ip_config_name = f'{self.prefix}-ip-config'
    self.nic_name = f'{self.prefix}-nic'
    self.nsg_name = f'{self.prefix}-nsg'
    self.vm_name = f'{self.prefix}-vm'

  def get_ip_address(self):
    return self.ip_address

  def deploy(self):
    # For the foundations of this code, wee:
    # https://learn.microsoft.com/en-us/azure/developer/python/sdk/examples/azure-sdk-example-virtual-machines?tabs=cmd#3-write-code-to-create-a-virtual-machine

    # Acquire a credential object using CLI-based authentication.
    credential = AzureCliCredential()
    
    # Obtain the management object for resources, using the credentials from the CLI login.
    resource_client = ResourceManagementClient(credential, self.subscription_id)
    
    # Provision the resource group.
    print(f'Provisioning resource group {self.resource_group_name} ({self.location} region)') 
    rg_result = resource_client.resource_groups.create_or_update(
      self.resource_group_name, {'location': self.location}
    )
    self._check_result('resource group', rg_result)
    
    # Obtain the management object for networks
    network_client = NetworkManagementClient(credential, self.subscription_id)
    
    # Provision the network security group with SSH access
    print(f'Provisioning network security group {self.nsg_name}')
    poller = network_client.network_security_groups.begin_create_or_update(
      self.resource_group_name,
      self.nsg_name,
      {
        'location': self.location,
        'properties': {
          'securityRules': [
            {
              'name': 'SSH',
              'properties': {
                'protocol': 'TCP',
                'sourcePortRange': '*',
                'destinationPortRange': '22',
                'sourceAddressPrefix': '*',
                'destinationAddressPrefix': '*',
                'access': 'Allow',
                'priority': 300,
                'direction': 'Inbound',
                'sourcePortRanges': [],
                'destinationPortRanges': [],
                'sourceAddressPrefixes': [],
                'destinationAddressPrefixes': []
              }
            }
          ]
        },
      },
    )
    
    nsg_result = poller.result()
    self._check_result('network security group', nsg_result)
    
    # Provision the virtual network
    print(f'Provisioning virtual network {self.vnet_name}')
    poller = network_client.virtual_networks.begin_create_or_update(
      self.resource_group_name,
      self.vnet_name,
      {
        'location': self.location,
        'addressSpace': {'addressPrefixes': ['10.0.0.0/16']},
      },
    )
    
    vnet_result = poller.result()
    self._check_result('virtual network', vnet_result)
    
    # Provision the subnet
    print(f'Provisioning subnet {self.vnet_name}')
    poller = network_client.subnets.begin_create_or_update(
      self.resource_group_name,
      self.vnet_name,
      self.subnet_name,
      {'addressPrefix': '10.0.0.0/24'},
    )
    subnet_result = poller.result()
    self._check_result('subnet', subnet_result)
     
    # Provision an IP address
    print(f'Provisioning public ip address {self.ip_name}')
    poller = network_client.public_ip_addresses.begin_create_or_update(
      self.resource_group_name,
      self.ip_name,
      {
        'location': self.location,
        'sku': {'name': 'Standard'},
        'publicIpAllocationMethod': 'Static',
        'publicIpAddressVersion': 'IPV4',
      },
    )
    
    ip_address_result = poller.result()
    self._check_result('ip address', ip_address_result)
    self.ip_address = ip_address_result.ip_address
    print(f'Public IP address: {self.ip_address}')
    print()
    
    # Provision the network interface client
    print(f'Provisioning network interface {self.nic_name}')
    poller = network_client.network_interfaces.begin_create_or_update(
      self.resource_group_name,
      self.nic_name,
      {
        'location': self.location,
        'ipConfigurations': [
          {
            'name': self.ip_config_name,
            'subnet': {'id': subnet_result.id},
            'publicIpAddress': {'id': ip_address_result.id},
          }
        ],
        'networkSecurityGroup': {'id': nsg_result.id},
      },
    )
    
    nic_result = poller.result()
    self._check_result('network interface', nic_result)
    
    # Obtain the management object for virtual machines
    self.compute_client = ComputeManagementClient(credential, self.subscription_id)
    
    # Provision the virtual machine
    print(f'Provisioning virtual machine {self.vm_name}')
    poller = self.compute_client.virtual_machines.begin_create_or_update(
      self.resource_group_name,
      self.vm_name,
      {
        'location': self.location,
        'storageProfile': {
          'imageReference': {
            'publisher': 'canonical',
            'offer': '0001-com-ubuntu-server-focal',
            'sku': '20_04-lts-gen2',
            'version': 'latest',
          }
        },
        'hardwareProfile': {'vm_size': 'Standard_D2s_v3'},
        'osProfile': {
          'computerName': self.vm_name,
          'adminUsername': self.username,
          'linuxConfiguration': {
            'disablePasswordAuthentication': True,
            'ssh': {
              'publicKeys': [
                {
                  'path': f'/home/{self.username}/.ssh/authorized_keys',
                  'keyData': f'{self.ssh_public_key}',
                },
              ]
            }
          }
        },
        'networkProfile': {
          'networkInterfaces': [
            {
              'id': nic_result.id,
            }
          ]
        },
      },
    )
    
    vm_result = poller.result()
    self._check_result('virtual machine', vm_result)
    print(f'All provisioning steps completed')    

  def check_status(self):
    vm_get_result = self.compute_client.virtual_machines.get(self.resource_group_name, self.vm_name, expand='instanceView')
    return vm_get_result.instance_view.statuses[1]

  def await_running(self):
    print('Checking virtual machine status')
    for i in range(5):
      status = self.check_status()
      print('Status: {}'.format(status.display_status))
      print('Status code: {}'.format(status.code))
      if status.code == 'PowerState/running':
        return True
      time.sleep(5)
    return False

class RemoteTrainer:
  ip_address = None
  ssh_key_file = None
  username = 'azure'

  def __init__(self, ip_address, ssh_key_file):
    self.ip_address = ip_address
    self.ssh_key_file = ssh_key_file

  def _check_run(self, command, result):
    if result.ok:
      print('Executing {} succeeded'.format(command))
    else:
      print('Executing {} failed'.format(command))
      exit()

  def execute(self):  
    print(f'Logging in and executing functions on the host')
    success = False
    for i in range(5):
      if success:
        break
      try:
        print('Connection attempt {}'.format(i + 1))
        with Connection(self.ip_address, user=self.username, connect_kwargs={'key_filename': self.ssh_key_file}) as connection:
          connection.run('git clone https://github.com/llewelld/minGPT.git ~/minGPT')
          with connection.cd('~/minGPT'):
            success = True
            result = connection.run('git checkout gpu-tests')
            self._check_run('git checkout', result)
            result = connection.run('SLURM_ARRAY_TASK_ID=1 ./scripts/azure-batch.sh')
            self._check_run('mingpt', result)
      except (NoValidConnectionsError, TimeoutError):
        print('Connection attempt {} error'.format(i + 1))
        time.sleep(5)

# Deploy a remote Virtual Machine on Azure

ssh_key_file = os.path.expanduser('~/.ssh/azure.pem')
name = RandomName('ml-test', 5).get()
print(f'Provisioning: {name}')
print()

# Retrieve subscription ID from environment variable.
subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
 
virtual_machine = VirtualMachine(name, ssh_key_file, subscription_id)
virtual_machine.deploy()

# Wait for the machine to come up
virtual_machine.await_running()

# Execute the training script on the remote virtual machine
trainer = RemoteTrainer(virtual_machine.get_ip_address(), ssh_key_file)
trainer.execute()

