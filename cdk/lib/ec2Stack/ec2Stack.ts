import { Stack, StackProps, RemovalPolicy, aws_s3 as s3, Fn } from 'aws-cdk-lib';
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

    // Network setting for EC2
    const defaultVpc = ec2.Vpc.fromLookup(this, 'VPC', {
      isDefault: true,
    });

    const chatbotAppSecurityGroup = new ec2.SecurityGroup(this, 'chatbotAppSecurityGroup', {
      vpc: defaultVpc,
    });
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'httpIpv4',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      'sshIpv4',
    );
    chatbotAppSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8501),
      'streamlitIpv4',
    );

    // set AMI
    const machineImage = ec2.MachineImage.fromSsmParameter(
      '/aws/service/canonical/ubuntu/server/focal/stable/current/amd64/hvm/ebs-gp2/ami-id'
    );
    
    // Get OpenSearch domain endpoint from SSM Parameter Store
    const opensearchDomainEndpoint = ssm.StringParameter.fromStringParameterName(
      this,
      'OpensearchDomainEndpoint',
      'opensearch_domain_endpoint'
    );
    
    // Get OpenSearch user ID from SSM Parameter Store
    const opensearchUserId = ssm.StringParameter.fromStringParameterName(
      this,
      'OpensearchUserId',
      'opensearch_user_id'
    );
    
    // Get OpenSearch user password from Secrets Manager
    const opensearchUserPassword = secretsmanager.Secret.fromSecretNameV2(
      this,
      'OpensearchUserPassword',
      'opensearch_user_password'
    );
    
    // set User Data
    const userData = ec2.UserData.forLinux();
    const userDataScript = fs.readFileSync(path.join(__dirname, 'userdata.sh'), 'utf8');
    userData.addCommands(userDataScript);
    
    // Add commands to create .env file with OpenSearch values
    userData.addCommands(
      'mkdir -p /home/ubuntu/finops-demo/application',
      'cat > /home/ubuntu/finops-demo/application/.env << EOF',
      'REGION=us-west-2',
      '',
      'SONNET=anthropic.claude-3-5-sonnet-20241022-v2:0',
      'HAIKU=anthropic.claude-3-haiku-20240307-v1:0',
      'NOVA_PRO=amazon.nova-pro-v1:0',
      '',
      `OPENSEARCH_DOMAIN_ENDPOINT=$(aws ssm get-parameter --name opensearch_domain_endpoint --query "Parameter.Value" --output text)`,
      `OPENSEARCH_USER_ID=$(aws ssm get-parameter --name opensearch_user_id --query "Parameter.Value" --output text)`,
      `OPENSEARCH_USER_PASSWORD=$(aws secretsmanager get-secret-value --secret-id opensearch_user_password --query "SecretString" --output text | jq -r .pwkey)`,
      '',
      'TABLE_DESCRIPTION_INDEX=schema_description',
      'EXAMPLE_QUERIES_INDEX=sample_queries',
      '',
      'DIALECT=amazon_athena',
      'ATHENA_REGION=us-east-1',
      'ATHENA_RESULTS_S3_BUCKET=<TO_BE_UPDATED>',
      'DATABASE_NAME=<TO_BE_UPDATED>',
      'EOF',
      'chown ubuntu:ubuntu /home/ubuntu/finops-demo/application/.env'
    );
    
    // EC2 instance
    const chatbotAppInstance = new ec2.Instance(this, 'chatbotAppInstance', {
      instanceType: new ec2.InstanceType('m5.large'),
      machineImage: machineImage,
      vpc: defaultVpc,
      securityGroup: chatbotAppSecurityGroup,
      role: instanceRole,
      userData: userData,
    });

    new cdk.CfnOutput(this, 'chatbotAppUrl', {
      value: `http://${chatbotAppInstance.instancePublicIp}/`,
      description: 'The URL of AWS FinOps Chatbot instance',
      exportName: 'chatbotAppUrl',
    });
  }
}
