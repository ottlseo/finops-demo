import { Stack, StackProps, Fn } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as fs from 'fs';
import * as path from 'path';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

export class EC2Stack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // IAM Role to access EC2
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
      ],
    });

    // Add permissions to access SSM Parameter Store and Secrets Manager
    instanceRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ssm:GetParameter',
        'secretsmanager:GetSecretValue'
      ],
      resources: ['*'],
    }));

    // Create a new VPC with public and private subnets
    const vpc = new ec2.Vpc(this, 'FinOpsVPC', {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        }
      ],
    });

    // Create security group for the EC2 instance
    const chatbotAppSecurityGroup = new ec2.SecurityGroup(this, 'chatbotAppSecurityGroup', {
      vpc: vpc,
      description: 'Security group for FinOps Chatbot EC2 instance',
      allowAllOutbound: true,
    });
    
    // Add inbound rules
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'Allow HTTP access from anywhere',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'Allow SSH access from anywhere',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8501),
      'Allow Streamlit access from anywhere',
    );

    // set AMI
    const machineImage = ec2.MachineImage.fromSsmParameter(
      '/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id'
    );
    
    // set User Data
    const userData = ec2.UserData.forLinux();
    const userDataScript = fs.readFileSync(path.join(__dirname, 'userdata.sh'), 'utf8');
    userData.addCommands(userDataScript);
    
    // Add commands to create .env file with OpenSearch values
    userData.addCommands(
      'OPENSEARCH_DOMAIN_ENDPOINT=$(aws ssm get-parameter --name opensearch_domain_endpoint --query "Parameter.Value" --output text --region us-west-2 || echo "")',
      'OPENSEARCH_DOMAIN_ENDPOINT="https://$OPENSEARCH_DOMAIN_ENDPOINT"',
      'OPENSEARCH_USER_ID=$(aws ssm get-parameter --name opensearch_user_id --query "Parameter.Value" --output text --region us-west-2 || echo "raguser")',
      'OPENSEARCH_USER_PASSWORD=$(aws secretsmanager get-secret-value --secret-id opensearch_user_password --query "SecretString" --output text --region us-west-2 | jq -r .pwkey || echo "MarsEarth1!")',
      '',
      'DATABASE_NAME=$(aws ssm get-parameter --name database_name --query "Parameter.Value" --output text --region us-west-2 || echo "cur")',
      'ATHENA_REGION=$(aws ssm get-parameter --name athena_region --query "Parameter.Value" --output text --region us-west-2 || echo "us-east-1")',
      'ATHENA_RESULTS_S3_BUCKET=$(aws ssm get-parameter --name athena_results_s3_bucket --query "Parameter.Value" --output text --region us-west-2 || echo "")',
      'DATABASE_PORT=$(aws ssm get-parameter --name database_port --query "Parameter.Value" --output text --region us-west-2 || echo "443")',
      '',
      'mkdir -p /home/ubuntu/finops-demo/application',
      'cat > /home/ubuntu/finops-demo/application/.env << EOF',
      '',
      'REGION=us-west-2',
      '',
      'SONNET=anthropic.claude-3-5-sonnet-20241022-v2:0',
      'HAIKU=anthropic.claude-3-haiku-20240307-v1:0',
      'NOVA_PRO=amazon.nova-pro-v1:0',
      '',
      'OPENSEARCH_DOMAIN_ENDPOINT=${OPENSEARCH_DOMAIN_ENDPOINT}',
      'OPENSEARCH_USER_ID=${OPENSEARCH_USER_ID}',
      'OPENSEARCH_USER_PASSWORD=${OPENSEARCH_USER_PASSWORD}',
      '',
      'TABLE_DESCRIPTION_INDEX=schema_description',
      'EXAMPLE_QUERIES_INDEX=sample_queries',
      '',
      'DIALECT=amazon_athena',
      'DATABASE_NAME=${DATABASE_NAME}',
      'ATHENA_REGION=${ATHENA_REGION}',
      'ATHENA_RESULTS_S3_BUCKET=${ATHENA_RESULTS_S3_BUCKET}',
      'DATABASE_PORT=${DATABASE_PORT}',
      '',
      'EOF',
      'chown ubuntu:ubuntu /home/ubuntu/finops-demo/application/.env'
    );
    
    // EC2 instance
    const chatbotAppInstance = new ec2.Instance(this, 'chatbotAppInstance', {
      instanceType: new ec2.InstanceType('m5.large'),
      machineImage: machineImage,
      vpc: vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC, // Place in public subnet to get public IP
      },
      securityGroup: chatbotAppSecurityGroup,
      role: instanceRole,
      userData: userData,
      blockDevices: [
        {
          deviceName: '/dev/sda1',
          volume: ec2.BlockDeviceVolume.ebs(30, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    // Output the EC2 instance public IP and DNS
    new cdk.CfnOutput(this, 'ChatbotAppUrl', {
      value: `http://${chatbotAppInstance.instancePublicIp}/`,
      description: 'The URL of AWS FinOps Chatbot instance - Please wait for 5 minutes from now',
      exportName: 'ChatbotAppUrl',
    });
    
    new cdk.CfnOutput(this, 'ChatbotAppPublicDns', {
      value: chatbotAppInstance.instancePublicDnsName,
      description: 'Public DNS of the EC2 instance',
      exportName: 'ChatbotAppPublicDns',
    });
    
    new cdk.CfnOutput(this, 'VpcId', {
      value: vpc.vpcId,
      description: 'The ID of the VPC',
      exportName: 'FinOpsVpcId',
    });
  }
}
